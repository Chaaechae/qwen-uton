"""
Utonia V1m3-B: 2D-3D alignment (Qwen3.5 ViT teacher) + EMA-fixed 3D self-SSL.

Fixes the two critical bugs in `utonia_v1m2_qwen3_5_distill`:
  (1) `teacher.{mask,unmask}_head` were never initialized from the student
      counterparts → sinkhorn-knopp targets came from random prototypes.
  (2) `after_step` was an empty `pass` → teacher backbone & heads never EMA-
      followed the student, so the SSL targets stayed random forever.

The 2D alignment path (Qwen ViT → patch_proj → cosine sim) is unchanged from
v1m2 — Qwen is loaded as `self.enc2d_model`, frozen, and never participates
in EMA.

Recommended usage: pair with lower SSL loss weights (e.g. mask/roll=1/16,
unmask=1/8, enc2d=3/4) since the Qwen flavor of 2D supervision is more
semantically loaded than the original DINOv2 teacher.
"""

from itertools import chain
from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_
from transformers import AutoModel, AutoModelForImageTextToText
from copy import deepcopy

import pointops
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import (
    offset2batch,
    offset2bincount,
    batch2offset,
    bincount2offset,
)
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler


class OnlineCluster(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=4096,
        embed_channels=512,
        num_prototypes=4096,
        enable_mlp=True,
    ):
        super().__init__()
        if enable_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, embed_channels),
            )
        self.apply(self._init_weights)
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = torch.nn.utils.parametrizations.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False
        else:
            self.prototype = torch.nn.utils.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        if hasattr(self, "mlp"):
            feat = self.mlp(feat)
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)
        return self.prototype(feat)


@MODELS.register_module("Utonia-v1m3b_qwen3_5_distill_ema")
class UtoniaQwen3_5DistillEMA(PointModel):
    """Distillation module with restored DINO-style EMA on the 3D side."""

    def __init__(
        self,
        image_weight_name,
        image_weight_path,
        backbone_s,
        backbone_t,
        head_in_channels_s,
        head_in_channels_t,
        backbone_out_channels,
        patch_w,
        patch_h,
        head_hidden_channels=4096,
        head_embed_channels=512,
        head_num_prototypes=4096,
        enc2d_head_in_channels=1024,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_size_start=5,
        mask_size_base=20,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=1 / 16,
        roll_mask_loss_weight=1 / 16,
        unmask_loss_weight=1 / 8,
        enc2d_loss_weight=3 / 4,
        momentum_base=0.996,
        momentum_final=1,
        match_max_k=8,
        match_max_r=0.08,
        up_cast_level=2,
        teacher_pretrained_path=None,
        student_pretrained_path=None,
        enc2d_upcast_level=4,
        enc2d_cos_shift=True,
        ema_teacher_backbone=True,
    ):
        super().__init__()
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight
        self.enc2d_loss_weight = enc2d_loss_weight

        self.num_global_view = num_global_view
        self.num_local_view = num_local_view

        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None

        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None

        self.mask_jitter = mask_jitter

        self.teacher_temp = teacher_temp_start
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_warmup_ratio = teacher_temp_warmup_ratio
        self.teacher_temp_scheduler = None
        self.student_temp = student_temp

        self.momentum = momentum_base
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.momentum_scheduler = None

        self.match_max_k = match_max_k
        self.match_max_r = match_max_r
        self.up_cast_level = up_cast_level
        self.enc2d_upcast_level = enc2d_upcast_level
        self.ema_teacher_backbone = ema_teacher_backbone

        assert (
            unmask_loss_weight
            + mask_loss_weight
            + roll_mask_loss_weight
            + enc2d_loss_weight
            > 0
        )
        assert num_global_view > 1 or roll_mask_loss_weight == 0
        assert num_global_view in (1, 2)

        student_model_dict, teacher_model_dict = {}, {}
        teacher_custom = teacher_custom or {}
        student_backbone = build_model(backbone_s)
        backbone_t.update(teacher_custom)
        teacher_backbone = build_model(backbone_t)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        if self.enc2d_loss_weight > 0:
            self.patch_h = patch_h
            self.patch_w = patch_w
            self.image_weight_name = image_weight_name
            self.enc2d_model = self.load_enc2d(image_weight_name, image_weight_path)
            self.enc2d_model.requires_grad_(False)
            self._num_channels = enc2d_head_in_channels
            self.patch_proj = nn.Linear(backbone_out_channels, self._num_channels)

        head_t = partial(
            OnlineCluster,
            in_channels=head_in_channels_t,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        head_s = partial(
            OnlineCluster,
            in_channels=head_in_channels_s,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head_s()
            teacher_model_dict["mask_head"] = head_t()
        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head_s()
            teacher_model_dict["unmask_head"] = head_t()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        if student_pretrained_path is not None:
            self.student = self._load_backbone_warmstart(
                self.student, student_pretrained_path
            )
        if teacher_pretrained_path is not None:
            self.teacher = self._load_backbone_warmstart(
                self.teacher, teacher_pretrained_path
            )

        # *** FIX (1) ***: at construction time, copy student-head weights into
        # the matching teacher-head module. Without this, the teacher head
        # starts at a *different* random init than the student head and the
        # sinkhorn target distribution is meaningless. v1m2 forgot this.
        for k in list(self.student.keys()):
            if "head" in k and k in self.teacher:
                self.teacher[k].load_state_dict(self.student[k].state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.enc2d_cos_shift = enc2d_cos_shift

    def load_enc2d(self, model_name, model_weight):
        if "qwen3_5" in model_name.lower() or "qwen3.5" in model_name.lower():
            full = AutoModelForImageTextToText.from_pretrained(
                model_weight, trust_remote_code=True
            )
            visual = full.model.visual
            del full
            return visual.eval()
        model = AutoModel.from_pretrained(model_weight, trust_remote_code=True)
        return model.eval()

    def _load_backbone_warmstart(self, module_dict, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unexpected ckpt format at {path}: {type(state_dict)}")
        # Robust dispatch — check ALL keys, not just the first.
        if any(k.startswith("module.student.backbone.") for k in state_dict):
            stripped = {
                k.replace("module.student.backbone.", ""): v
                for k, v in state_dict.items()
                if "module.student.backbone." in k
            }
        else:
            stripped = state_dict
        prefixed = {f"backbone.{k}": v for k, v in stripped.items()}
        info = module_dict.load_state_dict(prefixed, strict=False)
        print(
            f"[v1m3b warm-start] {path}: loaded={len(prefixed) - len(info[1])}, "
            f"missing={len(info[0])}, unexpected={len(info[1])}"
        )
        return module_dict

    @torch.no_grad()
    def ENC2D_forward(self, x):
        B, C, H_pix, W_pix = x.shape
        T = self.enc2d_model.config.temporal_patch_size
        P = self.enc2d_model.config.patch_size
        assert H_pix % P == 0 and W_pix % P == 0
        h, w = H_pix // P, W_pix // P
        assert h == self.patch_h and w == self.patch_w

        x_t = x.unsqueeze(1).repeat(1, T, 1, 1, 1)
        patches = x_t.view(B, T, C, h, P, w, P)
        patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        patches = patches.view(B * h * w, T * C * P * P)

        grid_thw = torch.tensor(
            [[1, h, w]] * B, device=x.device, dtype=torch.long
        )
        hidden = self.enc2d_model.patch_embed(patches)
        rotary_pos_emb = self.enc2d_model.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.enc2d_model.blocks:
            hidden = blk(
                hidden,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return hidden.view(B, h * w, -1)

    def before_train(self):
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step
        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step
        self.teacher_temp_scheduler = CosineScheduler(
            start_value=self.teacher_temp_start,
            base_value=self.teacher_temp_base,
            final_value=self.teacher_temp_base,
            warmup_iters=int(total_steps * self.teacher_temp_warmup_ratio),
            total_iters=total_steps,
        )
        self.teacher_temp_scheduler.iter = curr_step
        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = curr_step

    def before_step(self):
        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.teacher_temp = self.teacher_temp_scheduler.step()
        self.momentum = self.momentum_scheduler.step()

        if self.trainer.writer is not None:
            w = self.trainer.writer
            w.add_scalar("params/mask_size", self.mask_size, self.mask_size_scheduler.iter)
            w.add_scalar("params/mask_ratio", self.mask_ratio, self.mask_ratio_scheduler.iter)
            w.add_scalar("params/teacher_temp", self.teacher_temp, self.teacher_temp_scheduler.iter)
            w.add_scalar("params/momentum", self.momentum, self.momentum_scheduler.iter)

    def after_step(self):
        # *** FIX (2) ***: DINO-style EMA — teacher follows student. The Qwen
        # ViT (`self.enc2d_model`) lives outside `self.student` / `self.teacher`
        # so it's never touched here; it stays frozen.
        with torch.no_grad():
            m = self.momentum
            if self.ema_teacher_backbone:
                s = list(self.student.backbone.parameters())
                t = list(self.teacher.backbone.parameters())
                torch._foreach_mul_(t, m)
                torch._foreach_add_(t, s, alpha=1 - m)

            s_heads = [p for n, p in self.student.named_parameters() if "head" in n]
            t_heads = [p for n, p in self.teacher.named_parameters() if "head" in n]
            if s_heads:
                torch._foreach_mul_(t_heads, m)
                torch._foreach_add_(t_heads, s_heads, alpha=1 - m)

    @staticmethod
    def sinkhorn_knopp(feat, temp, num_iter=3):
        feat = feat.float()
        q = torch.exp(feat / temp).t()
        n = sum(all_gather(q.shape[1]))
        k = q.shape[0]
        sum_q = q.sum()
        if get_world_size() > 1:
            dist.all_reduce(sum_q)
        q = q / sum_q
        for _ in range(num_iter):
            q_row_sum = q.sum(dim=1, keepdim=True)
            if get_world_size() > 1:
                dist.all_reduce(q_row_sum)
            q = q / q_row_sum / k
            q = q / q.sum(dim=0, keepdim=True) / n
        q *= n
        return q.t()

    def generate_mask(self, coord, offset, grid_size):
        batch = offset2batch(offset)
        mask_size = self.mask_size * grid_size
        mask_ratio = self.mask_ratio
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, _ = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * mask_ratio)
        patch_index = torch.randperm(patch_num, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    @torch.no_grad()
    def match_neighbour(self, view1_coord, view1_offset, view2_coord, view2_offset):
        index2, distance = pointops.knn_query(
            1,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index1 = torch.arange(
            index2.shape[0], device=index2.device, dtype=torch.long
        ).unsqueeze(-1)
        index = torch.cat([index1, index2], dim=-1)[
            distance.squeeze(-1) < self.match_max_r
        ]
        return index

    @torch.no_grad()
    def roll_point(self, point):
        n = self.num_global_view
        bs = len(point.offset) // self.num_global_view
        data_dict = {}
        for key in point.keys():
            if key in ["feat", "coord", "origin_coord", "batch"]:
                value = point[key].split(offset2bincount(point.offset).tolist())
                value = chain(*[value[n * b : n * (b + 1)][::-1] for b in range(bs)])
                if key == "batch":
                    value = [torch.ones_like(v) * i for i, v in enumerate(value)]
                data_dict[key] = torch.cat(list(value), dim=0)
        return Point(data_dict)

    def up_cast(self, point, upcast_level=None):
        if upcast_level is None:
            upcast_level = self.up_cast_level
        for _ in range(upcast_level):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    @staticmethod
    def pool_corr(point, correspondence):
        inverse_list, idx_ptr_list = [], []
        point_feat = dict(offset=point.offset, feat=point.feat)
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse_list.append(point.pop("pooling_inverse"))
            idx_ptr_list.append(point.pop("idx_ptr"))
            point = parent
        inverse_list.reverse()
        idx_ptr_list.reverse()
        for inverse, idx_ptr in zip(inverse_list, idx_ptr_list):
            _, indices = torch.sort(inverse)
            img_num = correspondence.shape[1]
            if img_num == 0:
                correspondence = -torch.ones((idx_ptr.shape[0] - 1, 0, 2)).cuda()
                continue
            correspondence_all = []
            for img_id in range(img_num):
                mask = torch.all(
                    correspondence[:, img_id] != torch.tensor([-1, -1]).cuda(),
                    dim=1,
                ).float()
                counts = torch_scatter.segment_csr(mask[indices], idx_ptr, reduce="sum")
                counts[counts == 0] = 100000
                correspondence_img = deepcopy(correspondence[:, img_id])
                correspondence_img[correspondence_img == -1] = 0
                mask_sum = torch_scatter.segment_csr(
                    correspondence_img[indices], idx_ptr, reduce="sum"
                )
                mask_sum = mask_sum / counts.unsqueeze(1)
                mask_sum[counts == 100000] = -1
                correspondence_all.append(mask_sum)
            correspondence = torch.stack(correspondence_all, dim=1)
        point_feat["correspondence"] = correspondence
        return Point(point_feat)

    def forward(self, data_dict, return_point=False):
        if return_point:
            point = self.teacher.backbone(data_dict)
            for _ in range(self.up_cast_level):
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return dict(point=point)

        with torch.no_grad():
            global_point = Point(
                feat=data_dict["global_feat_full"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )
            global_mask, _ = self.generate_mask(
                global_point.coord, global_point.offset, global_point.grid_size
            )
            mask_global_coord = global_point.coord.clone().detach()
            if self.mask_jitter is not None:
                mask_global_coord[global_mask] += torch.clip(
                    torch.randn_like(mask_global_coord[global_mask]).mul(
                        self.mask_jitter * data_dict["grid_size"][0]
                    ),
                    max=(self.mask_jitter * data_dict["grid_size"][0]) * 2,
                )

            mask_global_point = Point(
                feat=data_dict["global_feat"],
                coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"],
                mask=global_mask,
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )
            major_view_correspondence = data_dict["global_correspondence"]

            local_point = Point(
                feat=data_dict["local_feat"],
                coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"],
                offset=data_dict["local_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            result_dict = dict(loss=[])
            global_point_ = self.teacher.backbone(global_point)
            global_point_ = self.up_cast(global_point_)
            if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
                global_point_.feat = self.teacher.mask_head(global_point_.feat)
            elif self.unmask_loss_weight > 0:
                global_point_.feat = self.teacher.unmask_head(global_point_.feat)
            # If both SSL groups are off, leave global_point_.feat as raw backbone feat;
            # it won't be used (no SSL loss path will read it).

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)
            mask_pred_sim = self.student.mask_head(mask_global_point_.feat)

            if self.mask_loss_weight > 0:
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    mask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )
                mask_loss = -torch.sum(
                    mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                mask_loss = torch_scatter.segment_coo(
                    mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                roll_global_point_ = self.roll_point(global_point_)
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                    roll_mask_target_sim = self.sinkhorn_knopp(
                        roll_global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )
                roll_mask_loss = -torch.sum(
                    roll_mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(roll_mask_loss * self.roll_mask_loss_weight)

        if self.unmask_loss_weight > 0:
            local_point_ = self.student.backbone(local_point)
            local_point_ = self.up_cast(local_point_)
            unmask_pred_sim = self.student.unmask_head(local_point_.feat)
            with torch.no_grad():
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal_view_mask],
                    batch2offset(principal_view_batch),
                )
                unmask_target_sim = self.sinkhorn_knopp(
                    global_point_.feat[principal_view_mask][match_index[:, 1]],
                    self.teacher_temp,
                )
            unmask_loss = -torch.sum(
                unmask_target_sim
                * F.log_softmax(
                    unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            unmask_loss = torch_scatter.segment_coo(
                unmask_loss,
                index=local_point_.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)

        if self.enc2d_loss_weight > 0:
            if self.mask_loss_weight == 0 and self.roll_mask_loss_weight == 0:
                mask_global_point_ = self.student.backbone(mask_global_point)
                mask_global_point_ = self.up_cast(mask_global_point_)
            mask_global_point_enc2d = self.up_cast(
                mask_global_point_,
                upcast_level=self.enc2d_upcast_level - self.up_cast_level,
            )
            to_feature = self.pool_corr(
                mask_global_point_enc2d, major_view_correspondence
            )
            data_dict_global_offset = torch.cat(
                [torch.tensor([0]).cuda(), to_feature["offset"]], dim=0
            )
            enc2d_count = (
                data_dict_global_offset[
                    1 : len(data_dict_global_offset) : self.num_global_view
                ]
                - data_dict_global_offset[
                    0 : len(data_dict_global_offset) - 1 : self.num_global_view
                ]
            )
            enc2d_offset = torch.cat(
                [torch.tensor([0]).cuda(), torch.cumsum(enc2d_count, dim=0)]
            )
            enc2d_mask = torch.cat(
                [
                    torch.arange(0, c, device=enc2d_count.device)
                    + data_dict_global_offset[i * self.num_global_view]
                    for i, c in enumerate(enc2d_count)
                ],
                dim=0,
            )

            offset_points_3d = enc2d_offset[1:]
            batch_points_3d = offset2batch(offset_points_3d)
            imgs = data_dict["images"]
            feature3d = to_feature["feat"][enc2d_mask]
            correspondence = to_feature["correspondence"][enc2d_mask]
            mask = torch.any(
                correspondence != torch.tensor([-1, -1]).cuda(), dim=2
            )
            valid_index = torch.where(mask)

            bincount_img_num = data_dict["img_num"]
            offset_img_num = bincount2offset(bincount_img_num)
            total_img_num = offset_img_num[-1]

            if total_img_num > 0:
                with torch.no_grad():
                    feature2d = self.ENC2D_forward(imgs)
                    feature2d = feature2d.contiguous().view(-1, feature2d.shape[-1])

                offset_img_num = torch.cat([torch.tensor([0]).cuda(), offset_img_num])[:-1]
                batch_index = batch_points_3d[valid_index[0]]
                batch_img_num = offset_img_num[batch_index]
                feature3d_pixel = feature3d[valid_index[0]]

                feature_index = torch.cat(
                    [
                        batch_img_num.unsqueeze(-1),
                        valid_index[1].unsqueeze(-1),
                        correspondence[valid_index],
                    ],
                    dim=-1,
                ).long()
                feature_index = (
                    feature_index[:, 0] * self.patch_h * self.patch_w
                    + feature_index[:, 1] * self.patch_h * self.patch_w
                    + feature_index[:, 2] * self.patch_w
                    + feature_index[:, 3]
                )
                feature3d_pixel = torch_scatter.scatter_mean(
                    feature3d_pixel, feature_index, dim=0, dim_size=feature2d.shape[0]
                )
                feature3d_pixel = self.patch_proj(feature3d_pixel)
                feature_index = torch.unique(feature_index)
                feature2d_sel = feature2d[feature_index]
                feature3d_sel = feature3d_pixel[feature_index]

                if self.enc2d_cos_shift:
                    feature2d_sel = feature2d_sel - feature2d_sel.mean(dim=-1, keepdim=True)
                    feature3d_sel = feature3d_sel - feature3d_sel.mean(dim=-1, keepdim=True)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                loss = (1 - cos(feature2d_sel, feature3d_sel)).mean() * 10
                result_dict["enc2d_loss"] = loss
                result_dict["loss"].append(loss * self.enc2d_loss_weight)
            elif (
                self.mask_loss_weight
                + self.unmask_loss_weight
                + self.roll_mask_loss_weight
                > 0
            ):
                ssl_w = (
                    self.mask_loss_weight
                    + self.unmask_loss_weight
                    + self.roll_mask_loss_weight
                )
                result_ssl_loss = sum(result_dict["loss"]) / ssl_w
                result_dict["enc2d_loss"] = result_ssl_loss
                result_dict["loss"].append(result_ssl_loss * self.enc2d_loss_weight)

        if not result_dict["loss"]:
            zero = next(self.student.backbone.parameters()).sum() * 0.0
            result_dict["loss"] = zero
        else:
            result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            for k, v in list(result_dict.items()):
                if torch.is_tensor(v):
                    dist.all_reduce(v, op=dist.ReduceOp.AVG)
        return result_dict
