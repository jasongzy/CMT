import torch
import torch.nn.functional as F
from models.head.point_transformer import PointTransformerLayer
from models.head.PointSIFT.sift import PointSIFT_module
from models.head.scf import SCF_OE, SCF_Module
from pointnet2.utils import pointnet2_utils
from pointnet2.utils import pytorch_utils as pt_utils
from torch import nn


class BaseXCorr(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        self.mlp = pt_utils.SharedMLP([in_channel, hidden_channel, hidden_channel, hidden_channel], bn=True)
        self.fea_layer = (
            pt_utils.Seq(hidden_channel).conv1d(hidden_channel, bn=True).conv1d(out_channel, activation=None)
        )


class P2B_XCorr(BaseXCorr):
    def __init__(self, feature_channel, hidden_channel, out_channel):
        mlp_in_channel = feature_channel + 4
        super().__init__(mlp_in_channel, hidden_channel, out_channel)

    def forward(self, template_feature, search_feature, template_xyz):
        """

        :param template_feature: B,f,M
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :return:
        """
        B = template_feature.size(0)
        f = template_feature.size(1)
        n1 = template_feature.size(2)
        n2 = search_feature.size(2)
        final_out_cla = self.cosine(
            template_feature.unsqueeze(-1).expand(B, f, n1, n2), search_feature.unsqueeze(2).expand(B, f, n1, n2)
        )  # B,n1,n2

        fusion_feature = torch.cat(
            (final_out_cla.unsqueeze(1), template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B, 3, n1, n2)),
            dim=1,
        )  # B,1+3,n1,n2

        fusion_feature = torch.cat(
            (fusion_feature, template_feature.unsqueeze(-1).expand(B, f, n1, n2)), dim=1
        )  # B,1+3+f,n1,n2

        fusion_feature = self.mlp(fusion_feature)

        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])  # B, f, 1, n2
        fusion_feature = fusion_feature.squeeze(2)  # B, f, n2
        fusion_feature = self.fea_layer(fusion_feature)

        return fusion_feature


class BoxAwareXCorr(BaseXCorr):
    def __init__(
        self,
        feature_channel,
        hidden_channel,
        out_channel,
        k=8,
        use_search_bc=False,
        use_search_feature=False,
        bc_channel=9,
        scf_oe_use_xyz=True,
        neighbor_method="knn",
        kc=16,
        scf_match_method="swin",
        scf_swin_window_sizes=[],
        k2=4,
        attn_pe="xyz+bc",
        oe_radius=0.5,
    ):
        self.k = k
        self.use_search_bc = use_search_bc
        self.use_search_feature = use_search_feature
        mlp_in_channel = feature_channel
        if use_search_bc:
            mlp_in_channel += bc_channel
        if use_search_feature:
            mlp_in_channel += feature_channel
        mlp_in_channel += 3
        super(BoxAwareXCorr, self).__init__(mlp_in_channel, hidden_channel, out_channel)
        self.k2 = k2
        assert self.k2 <= self.k, "k2 should be smaller than k"
        self.attn_pe = attn_pe
        self.scf_oe = SCF_OE(scf_oe_use_xyz)
        self.scf = SCF_Module(neighbor_method, kc, scf_match_method, scf_swin_window_sizes)
        if attn_pe == "xyz":
            num_pos = 3
        elif attn_pe == "bc":
            num_pos = bc_channel
        elif attn_pe == "xyz+bc":
            num_pos = 3 + bc_channel
        else:
            raise ValueError(attn_pe)
        self.attn = PointTransformerLayer(
            dim=feature_channel, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_pos=num_pos
        )
        self.norm1 = nn.InstanceNorm1d(feature_channel)
        self.norm2 = nn.InstanceNorm1d(feature_channel)
        self.OE_module = PointSIFT_module(
            radius=oe_radius, output_channel=feature_channel, extra_input_channel=feature_channel
        )
        self.OE_module2 = PointSIFT_module(
            radius=oe_radius, output_channel=feature_channel, extra_input_channel=feature_channel
        )

    def forward(
        self, template_feature, search_feature, template_xyz, search_xyz=None, template_bc=None, search_bc=None
    ):
        """

        :param template_feature: B,f,M
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :param search_xyz: B.N,3
        :param template_bc: B,M,9
        :param search_bc: B.N,9
        :param args:
        :param kwargs:
        :return:
        """
        dist_matrix = torch.cdist(template_bc, search_bc)  # B, M, N
        template_xyz_feature_box = torch.cat(
            [template_xyz.transpose(1, 2).contiguous(), template_bc.transpose(1, 2).contiguous(), template_feature],
            dim=1,
        )
        # search_xyz_feature = torch.cat([search_xyz.transpose(1, 2).contiguous(), search_feature], dim=1)
        search_xyz_feature_box = torch.cat(
            [search_xyz.transpose(1, 2).contiguous(), search_bc.transpose(1, 2).contiguous(), search_feature], dim=1
        )
        template_scf = self.scf_oe(template_xyz)
        search_scf = self.scf_oe(search_xyz)
        template_xyz_feature_box = torch.cat([template_xyz_feature_box, template_scf], dim=1)
        search_xyz_feature_box = torch.cat([search_xyz_feature_box, search_scf], dim=1)
        top_k_nearest_idx_b = torch.argsort(dist_matrix, dim=1)[:, : 3 * self.k, :]
        top_k_nearest_idx_b = top_k_nearest_idx_b.transpose(1, 2).contiguous()
        dist_matrix_xyz = torch.cdist(search_xyz, template_xyz)
        mask = torch.full_like(dist_matrix_xyz, 1e5)
        mask.scatter_(2, top_k_nearest_idx_b, 1)
        dist_matrix_xyz = dist_matrix_xyz * mask
        top_k_nearest_idx_b = torch.argsort(dist_matrix_xyz.transpose(1, 2).contiguous(), dim=1)[:, : self.k, :]
        top_k_nearest_idx_b = top_k_nearest_idx_b.transpose(1, 2).contiguous()
        dist_matrix_scf = self.scf(search_xyz, template_xyz, top_k_nearest_idx_b)
        top_k_nearest_idx_scf = torch.argsort(dist_matrix_scf.transpose(1, 2).contiguous(), dim=1)[:, : self.k2, :]
        top_k_nearest_idx_scf = top_k_nearest_idx_scf.transpose(1, 2).contiguous()
        correspondences_b = pointnet2_utils.grouping_operation(template_xyz_feature_box, top_k_nearest_idx_scf.int())
        fusion_feature_raw = torch.cat([search_xyz_feature_box.unsqueeze(-1), correspondences_b], -1)
        fusion_feature_raw = fusion_feature_raw.permute(0, 2, 3, 1).contiguous()
        fusion_feature_raw = fusion_feature_raw[:, :, :, 12:]
        fusion_feature_raw = self.mlp(fusion_feature_raw.permute(0, 3, 1, 2).contiguous())
        fusion_feature_raw = fusion_feature_raw.permute(0, 2, 3, 1).contiguous()
        top_k_nearest_xyz = correspondences_b[:, 0:3]
        top_k_nearest_xyz = top_k_nearest_xyz.transpose(2, 3).transpose(1, 3).contiguous()
        search_template_xyz = torch.cat([search_xyz.unsqueeze(2), top_k_nearest_xyz], 2)
        top_k_nearest_bc = correspondences_b[:, 3:12]
        top_k_nearest_bc = top_k_nearest_bc.permute(0, 2, 3, 1).contiguous()
        search_template_bc = torch.cat([search_bc.unsqueeze(2), top_k_nearest_bc], 2)
        search_template_xyz_bc = torch.cat([search_template_xyz, search_template_bc], dim=-1)
        B, N, KPLUS1, D = fusion_feature_raw.shape
        fusion_feature_raw = fusion_feature_raw.view(-1, KPLUS1, D)
        # search_template_bc = search_template_bc.view(-1, KPLUS1, 9)
        search_template_xyz_bc = search_template_xyz_bc.view(-1, KPLUS1, 12)
        if self.attn_pe == "xyz":
            fusion_feature = self.attn(fusion_feature_raw, search_template_xyz)
        elif self.attn_pe == "bc":
            fusion_feature = self.attn(fusion_feature_raw, search_template_bc)
        elif self.attn_pe == "xyz+bc":
            fusion_feature = self.attn(fusion_feature_raw, search_template_xyz_bc)
        else:
            raise ValueError(self.attn_pe)
        fusion_feature = fusion_feature.view(B, N, KPLUS1, D)
        fusion_feature = fusion_feature.permute(0, 3, 1, 2).contiguous()
        fusion_feature = fusion_feature[:, :, :, 0]
        fusion_feature = fusion_feature + search_feature
        fusion_feature = self.norm1(fusion_feature.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        fusion_feature_final = self.fea_layer(fusion_feature)
        fusion_feature_final = fusion_feature_final + fusion_feature
        fusion_feature_final = (
            self.norm2(fusion_feature_final.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        )
        _, fusion_feature_contextual = self.OE_module(search_xyz, fusion_feature_final.transpose(1, 2).contiguous())
        fusion_feature_contextual = fusion_feature_contextual + fusion_feature_final
        _, fusion_feature_contextual_final = self.OE_module2(
            search_xyz, fusion_feature_contextual.transpose(1, 2).contiguous()
        )
        fusion_feature_contextual_final = fusion_feature_contextual_final + fusion_feature_contextual
        return fusion_feature_contextual_final
