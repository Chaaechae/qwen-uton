"""
Utonia V1m3-A: pure 2D-3D alignment (Qwen3.5 ViT teacher, no SSL).

Goal: train Utonia (PTv3) so that point features, after a learned projection
(`patch_proj`), land in the same manifold as Qwen3.5 vision-tower patch tokens.

Differences vs. utonia_v1m2_qwen3_5_distill:
  * No teacher PTv3 (no EMA, no momentum scheduler).
  * No mask / roll-mask / unmask SSL heads.
  * No sinkhorn-knopp, no per-step scheduler bookkeeping for SSL.
  * Forward returns only `enc2d_loss` (cosine alignment to Qwen patch tokens).

This is the most direct, minimum-surface implementation of the alignment goal:
gradient flows from Qwen feature targets → patch_proj → student PTv3 backbone.
Everything else from the Sonata/Concerto SSL recipe is removed.
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from transformers import AutoModel, AutoModelForImageTextToText

from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, bincount2offset
from pointcept.utils.comm import get_world_size


@MODELS.register_module("Utonia-v1m3a_qwen3_5_align_only")
class UtoniaQwen3_5AlignOnly(PointModel):
    def __init__(
        self,
        image_weight_name,
        image_weight_path,
        backbone,
        backbone_out_channels,
        patch_w,
        patch_h,
        enc2d_head_in_channels=1024,
        num_global_view=2,
        up_cast_level=0,
        enc2d_upcast_level=3,
        enc2d_cos_shift=True,
        student_pretrained_path=None,
    ):
        super().__init__()
        self.num_global_view = num_global_view
        self.up_cast_level = up_cast_level
        self.enc2d_upcast_level = enc2d_upcast_level
        self.enc2d_cos_shift = enc2d_cos_shift
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.image_weight_name = image_weight_name
        self._num_channels = enc2d_head_in_channels

        student_backbone = build_model(backbone)
        self.student = nn.ModuleDict({"backbone": student_backbone})

        self.enc2d_model = self.load_enc2d(image_weight_name, image_weight_path)
        self.enc2d_model.requires_grad_(False)
        self.patch_proj = nn.Linear(backbone_out_channels, self._num_channels)

        if student_pretrained_path is not None:
            self._warmstart_backbone(student_pretrained_path)

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

    def _warmstart_backbone(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if any(k.startswith("module.student.backbone.") for k in state_dict):
            stripped = {
                k.replace("module.student.backbone.", ""): v
                for k, v in state_dict.items()
                if "module.student.backbone." in k
            }
        else:
            stripped = state_dict
        prefixed = {f"backbone.{k}": v for k, v in stripped.items()}
        info = self.student.load_state_dict(prefixed, strict=False)
        print(
            f"[v1m3a warm-start] {path}: loaded={len(prefixed) - len(info[1])}, "
            f"missing={len(info[0])}, unexpected={len(info[1])}"
        )

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

        grid_thw = torch.tensor([[1, h, w]] * B, device=x.device, dtype=torch.long)
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
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def up_cast(self, point, upcast_level=None):
        if upcast_level is None:
            upcast_level = self.up_cast_level
        for _ in range(upcast_level):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
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
                counts = torch_scatter.segment_csr(
                    mask[indices], idx_ptr, reduce="sum"
                )
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
            point = self.student.backbone(data_dict)
            for _ in range(self.up_cast_level):
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return dict(point=point)

        global_point = Point(
            feat=data_dict["global_feat"],
            coord=data_dict["global_coord"],
            origin_coord=data_dict["global_origin_coord"],
            offset=data_dict["global_offset"],
            grid_size=data_dict["grid_size"][0],
        )
        result_dict = dict(loss=[])

        point_ = self.student.backbone(global_point)
        point_ = self.up_cast(point_)
        point_enc2d = self.up_cast(
            point_,
            upcast_level=self.enc2d_upcast_level - self.up_cast_level,
        )

        to_feature = self.pool_corr(point_enc2d, data_dict["global_correspondence"])
        data_dict_global_offset = torch.cat(
            [torch.tensor([0]).cuda(), to_feature["offset"]], dim=0
        )
        enc2d_count = (
            data_dict_global_offset[1 :: self.num_global_view]
            - data_dict_global_offset[: -1 : self.num_global_view]
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
        v0 = correspondence.shape[1]
        valid_mask = torch.any(
            correspondence != torch.tensor([-1, -1]).cuda(), dim=2
        )
        valid_index = torch.where(valid_mask)

        bincount_img_num = data_dict["img_num"]
        offset_img_num = bincount2offset(bincount_img_num)
        total_img_num = offset_img_num[-1]

        if total_img_num == 0:
            zero = feature3d.sum() * 0.0
            result_dict["enc2d_loss"] = zero
            result_dict["loss"] = zero
            return result_dict

        feature2d = self.ENC2D_forward(imgs).contiguous().view(-1, self._num_channels)

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
        result_dict["loss"] = loss

        if get_world_size() > 1:
            for k in result_dict:
                dist.all_reduce(result_dict[k], op=dist.ReduceOp.AVG)
        return result_dict
