import point
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# from scipy.optimize import linear_sum_assignment
# from motmetrics.lap import linear_sum_assignment
# from lapsolver import solve_dense as linear_sum_assignment
from lap import lapjv
from models.head.ipot import ipot_dist
from pointnet2.utils.pointnet2_utils import grouping_operation


class SCF_Module(pl.LightningModule):
    def __init__(self, neighbor_method="knn", K=16, match_method="swin", swin_window_sizes=[4, 3, 3]):
        super().__init__()
        self.K = K
        self.match_method = match_method
        self.neighbor_method = neighbor_method

        if neighbor_method == "cube":
            self.K = 8
            self.cube_radius = 1e8
            # swin_window_sizes = [3, 2]

        if match_method == "shift_min":
            self.diag_matrices = self.generate_diag_matrices()
        elif match_method == "swin":
            self.window_sizes = swin_window_sizes
            self.swin_masks = self.generate_swin_masks()

    def update_device(self):
        if self.match_method == "shift_min":
            self.diag_matrices = self.diag_matrices.to(self.device)
        elif self.match_method == "swin":
            for size, swin_mask in self.swin_masks.items():
                self.swin_masks[size] = swin_mask.to(self.device)

    def forward(self, xyz1, xyz2, xyz2_idx, polar_dist=True):
        """
        :param xyz1: B,N,3
        :param xyz2: B,M,3
        :param xyz2_idx: B,N,L
        :return: B,N,M
        """
        self.update_device()
        feature1 = self.get_local_contextual_feature(xyz1)  # B,N,K,3
        feature2 = self.get_local_contextual_feature(xyz2)  # B,M,K,3
        if polar_dist:
            feature_xyz = []
            for feature in (feature1, feature2):
                # feature[-1]: [relative_dis, angle_alpha(yaw), angle_beta(pitch)]
                r = feature[:, :, :, 0]
                yaw = feature[:, :, :, 1]
                pitch = feature[:, :, :, 2]
                pitch_cos = torch.cos(pitch)
                x = r * pitch_cos * torch.cos(yaw)
                y = r * pitch_cos * torch.sin(yaw)
                z = r * torch.sin(pitch)
                feature_xyz.append(torch.stack([x, y, z], dim=-1))
            feature1, feature2 = feature_xyz[0], feature_xyz[1]
        # dist_matrix = self.get_neighborhood_distance_lap_fast(feature1.detach().cpu(), feature2.detach().cpu(), xyz2_idx.detach().cpu())
        dist_matrix = self.get_neighborhood_distance(feature1, feature2, xyz2_idx)
        return dist_matrix

    def KNN(self, points_xyz):
        """
        :param points_xyz: B,N,3
        :return: B,N,K
        """
        assert points_xyz.shape[1] > self.K, "K is too large"
        dist_matrix = torch.cdist(points_xyz, points_xyz)  # B,N,N
        top_k_nearest_idx = torch.argsort(dist_matrix, dim=1)[:, 1 : self.K + 1, :]  # B,K,N
        top_k_nearest_idx = top_k_nearest_idx.transpose(1, 2).contiguous()  # B,N,K
        return top_k_nearest_idx

    def cube_neighbor(self, points_xyz, radius=1e8):
        """
        :param points_xyz: B,N,3
        :return: B,N,8
        """
        idx = torch.zeros((points_xyz.shape[0], points_xyz.shape[1], 8), dtype=torch.int32).cuda()
        point.select_cube(points_xyz.cuda(), idx, points_xyz.shape[0], points_xyz.shape[1], radius)
        return idx.long().to(self.device)

    @staticmethod
    def gather_neighbor(pc, neighbor_idx):
        """
        gather the coordinates or features of neighboring points

        :param pc: B,M,D,*
        :param neighbor_idx: B,N,K (K: index of M)
        :return: B,N,K,D,*
        """
        pc_shape = pc.shape
        if len(pc_shape) > 3:
            pc = pc.view(*pc_shape[0:2], -1)
        if pc.is_cuda:
            # B,D,M  B,N,K --> B,D,N,K
            neighbor = grouping_operation(pc.transpose(1, 2).contiguous(), neighbor_idx.int())
            neighbor = neighbor.permute(0, 2, 3, 1).contiguous()
        else:
            B, N = neighbor_idx.shape[0:2]
            neighbor = []
            for b in range(B):
                neighbor_xyz_n = []
                for n in range(N):
                    neighbor_xyz_n.append(pc[b, neighbor_idx[b, n, :], :])
                neighbor.append(torch.stack(neighbor_xyz_n, dim=0))
            neighbor = torch.stack(neighbor, dim=0)
        if len(pc_shape) > 3:
            neighbor = neighbor.view(*neighbor_idx.shape, *pc_shape[2:])
        return neighbor

    def relative_pos_transforming(self, points_xyz, neighbor_xyz):
        """
        :param points_xyz: B,N,3
        :param neighbor_xyz: B,N,K,3
        """
        points_xyz = points_xyz.repeat(self.K, 1, 1, 1)  # K,B,N,3
        points_xyz = points_xyz.permute(1, 2, 0, 3).contiguous()  # B,N,K,3
        relative_xyz = points_xyz - neighbor_xyz  # B,N,K,3
        # yaw
        relative_alpha = torch.atan2(relative_xyz[:, :, :, 1], relative_xyz[:, :, :, 0])  # B,N,K
        relative_xydis = torch.sqrt(torch.pow(relative_xyz[:, :, :, :2], 2).sum(axis=-1))  # B,N,K
        # pitch
        relative_beta = torch.atan2(relative_xyz[:, :, :, 2], relative_xydis)  # B,N,K
        relative_dis = torch.sqrt(torch.pow(relative_xyz, 2).sum(axis=-1))  # B,N,K
        return relative_dis, relative_alpha, relative_beta

    def local_polar_representation(self, points_xyz, neighbor_xyz):
        """
        :param points_xyz: B,N,3
        :param neighbor_xyz: B,N,K,3
        :return: B,N,
                relative_dis 1
                angle_alpha 1
                angle_beta 1
        """
        relative_dis, relative_alpha, relative_beta = self.relative_pos_transforming(points_xyz, neighbor_xyz)  # B,N,K
        # Local direction calculation (angle)
        # center-of-mass point
        neighbor_mean = neighbor_xyz.mean(axis=-2)  # B,N,3
        direction = points_xyz - neighbor_mean  # B,N,3
        direction = direction.repeat(self.K, 1, 1, 1)  # K,B,N,3
        direction = direction.permute(1, 2, 0, 3).contiguous()  # B,N,K,3
        direction_alpha = torch.atan2(direction[:, :, :, 1], direction[:, :, :, 0])  # B,N,K
        direction_xydis = torch.sqrt(torch.pow(direction[:, :, :, :2], 2).sum(axis=-1))  # B,N,K
        direction_beta = torch.atan2(direction[:, :, :, 2], direction_xydis)  # B,N,K
        # Polar angle updating
        angle_alpha = relative_alpha - direction_alpha  # B,N,K
        angle_beta = relative_beta - direction_beta  # B,N,K
        return relative_dis, angle_alpha, angle_beta

    def get_local_contextual_feature(self, points_xyz):
        """
        :param points_xyz: B,N,3
        :return: B,N,K,(1+1+1)
        """
        if self.neighbor_method == "knn":
            neighbor_idx = self.KNN(points_xyz)
        elif self.neighbor_method == "cube":
            neighbor_idx = self.cube_neighbor(points_xyz, radius=self.cube_radius)
        else:
            raise Exception("unknown neighbor_method")
        neighbor_xyz = self.gather_neighbor(points_xyz, neighbor_idx)  # B,N,K,3
        relative_dis, angle_alpha, angle_beta = self.local_polar_representation(points_xyz, neighbor_xyz)  # B,N,K
        local_contextual_feature = torch.stack([relative_dis, angle_alpha, angle_beta], dim=-1)  # B,N,K,3
        return local_contextual_feature

    def get_feature_distance_lap(self, dist_matrix):
        """
        :param dist_matrix: B,K,K
        :return: B
        """
        dist_batches = torch.zeros(dist_matrix.shape[0])
        for i, batch in enumerate(dist_matrix):
            # row_ind, col_ind = linear_sum_assignment(batch)
            # dist_batches.append(batch[row_ind, col_ind].sum().unsqueeze(0))
            cost, _, _ = lapjv(batch.numpy())
            dist_batches[i] = cost
        return dist_batches  # [B]

    def get_neighborhood_distance_lap(self, neighborhood1, neighborhood2, neighborhood2_idx):
        """
        ** Deprecated **
        
        :param neighborhood1: B,N,K,D
        :param neighborhood2: B,M,K,D
        :param neighborhood2_idx: B,N,L  indices that indicate a subset of neighborhood2 (L<=M)
        :return: B,N,M
        """
        dist_matrix = torch.full(
            [neighborhood1.shape[0], neighborhood1.shape[1], neighborhood2.shape[1]], float("inf")
        )  # B,N,M
        B = range(neighborhood1.shape[0])
        for b in B:
            for i, feature1 in enumerate(neighborhood1[b]):  # feature1: K,D
                neighborhood2_batch = neighborhood2[b][neighborhood2_idx[b, i, :]]  # L,K,D
                for j, feature2 in enumerate(neighborhood2_batch):  # feature2: K,D
                    feature_dist_matrix = torch.cdist(feature1.unsqueeze(0), feature2.unsqueeze(0)).squeeze(0)  # K,K
                    # row_ind, col_ind = linear_sum_assignment(feature_dist_matrix)
                    # cost = feature_dist_matrix[row_ind, col_ind].sum()
                    cost, _, _ = lapjv(feature_dist_matrix.numpy())
                    dist_matrix[b, i, neighborhood2_idx[b, i, j]] = cost
        return dist_matrix

    def get_neighborhood_distance_lap_fast(self, neighborhood1, neighborhood2, neighborhood2_idx):
        """
        ** Deprecated **

        :param neighborhood1: B,N,K,D
        :param neighborhood2: B,M,K,D
        :param neighborhood2_idx: B,N,L  indices that indicate a subset of neighborhood2 (L<=M)
        :return: B,N,M
        """
        B, N, K, D = neighborhood1.shape
        M = neighborhood2.shape[1]
        L = neighborhood2_idx.shape[2]
        dist_matrix_masked = torch.zeros([B, N, L])
        neighborhood1 = neighborhood1.transpose(0, 1).contiguous()  # N,B,K,D
        for n, feature1 in enumerate(neighborhood1):
            neighborhood2_masked = torch.zeros([B, L, K, D])
            for b in range(B):
                neighborhood2_masked[b] = neighborhood2[b, neighborhood2_idx[b, n, :], :, :]  # L,K,D
            neighborhood2_masked = neighborhood2_masked.transpose(0, 1).contiguous()  # L,B,K,D
            for l, feature2 in enumerate(neighborhood2_masked):
                dist_matrix = torch.cdist(feature1, feature2)  # B,K,K
                dist_matrix_masked[:, n, l] = self.get_feature_distance_lap(dist_matrix)
        dist_matrix = torch.full([B, N, M], float("inf"))
        # for b in range(B):
        #     for n in range(N):
        #         dist_matrix[b, n, neighborhood2_idx[b, n, :]] = dist_matrix_masked[b, n, :]
        dist_matrix.scatter_(2, neighborhood2_idx, dist_matrix_masked)
        return dist_matrix

    def get_neighborhood_distance(self, neighborhood1, neighborhood2, neighborhood2_idx):
        """
        :param neighborhood1: B,N,K,D
        :param neighborhood2: B,M,K,D
        :param neighborhood2_idx: B,N,L  indices that indicate a subset of neighborhood2 (L<=M)
        :return: B,N,M
        """
        B, N = neighborhood1.shape[0:2]
        M = neighborhood2.shape[1]
        neighborhood1, neighborhood2 = self.batchify(neighborhood1, neighborhood2, neighborhood2_idx)  # B*N*L,K,D
        feature_dist_matrix = torch.cdist(neighborhood1, neighborhood2)  # B*N*L,K,K
        if self.match_method == "lap":
            point_dist_matrix_masked = self.get_feature_distance_lap(feature_dist_matrix.detach().cpu())  # B*N*L
            point_dist_matrix_masked = point_dist_matrix_masked.to(self.device)
        elif self.match_method == "shift_min":
            point_dist_matrix_masked = self.get_feature_distance_shift_min(feature_dist_matrix)  # B*N*L
        elif self.match_method == "swin":
            point_dist_matrix_masked = self.get_feature_distance_swin(feature_dist_matrix)  # B*N*L
        elif self.match_method == "ipot":
            point_dist_matrix_masked = ipot_dist(feature_dist_matrix)  # B*N*L
        else:
            raise Exception("unknown match_method")
        point_dist_matrix_masked = point_dist_matrix_masked.view_as(neighborhood2_idx)  # B,N,L
        point_dist_matrix = torch.full([B, N, M], float("inf"), device=self.device)
        # for b in range(B):
        # for n in range(N):
        #     point_dist_matrix[b, n, neighborhood2_idx[b, n, :]] = point_dist_matrix_masked[b, n, :]
        point_dist_matrix.scatter_(2, neighborhood2_idx, point_dist_matrix_masked)
        return point_dist_matrix

    def batchify(self, neighborhood1, neighborhood2, neighborhood2_idx):
        """
        :param neighborhood1: B,N,K,D
        :param neighborhood2: B,M,K,D
        :param neighborhood2_idx: B,N,L (L: index of M)
        :return: (B*N*L,K,D)*2
        """
        B, N, K, D = neighborhood1.shape
        L = neighborhood2_idx.shape[-1]
        neighborhood1 = neighborhood1.repeat_interleave(L, 1)  # B,N*L,K,D
        neighborhood1 = neighborhood1.view(-1, K, D)  # B*N*L,K,D
        neighborhood2 = self.gather_neighbor(neighborhood2, neighborhood2_idx)  # B,N,L,K,D
        neighborhood2 = neighborhood2.view(-1, K, D)  # B*N*L,K,D
        return neighborhood1, neighborhood2

    def get_feature_distance_shift_min(self, feature_dist_matrix):
        """
        :param feature_dist_matrix: B,K,K
        :return: B
        """
        K = feature_dist_matrix.shape[1]
        dist = F.conv2d(feature_dist_matrix.unsqueeze(1), self.diag_matrices.unsqueeze(1)).squeeze()  # B,K//2*2+1
        dist = torch.min(dist, dim=1)[0]  # B
        return dist

    def generate_diag_matrices(self):
        """
        For shift_min (N = K//2)

        :return: (2N+1)*K*K
        """
        N = self.K // 2
        matrix_list = []
        for i in range(-N, N + 1):
            matrix = torch.ones([self.K, self.K], device=self.device)
            matrix = matrix.triu(i) - matrix.triu(i + 1)
            matrix /= matrix.sum()
            matrix_list.append(matrix)
        matrices = torch.stack(matrix_list, dim=0)
        return matrices

    def get_feature_distance_swin(self, feature_dist_matrix):
        """
        :param feature_dist_matrix: B,K,K
        :return: B
        """
        B, K = feature_dist_matrix.shape[0:2]
        assert K >= torch.Tensor(self.window_sizes).sum().item(), "swin_window_sizes too large"
        windows = []
        for size in self.window_sizes:
            windows.append(torch.eye(size, device=self.device))
        dists = torch.zeros([B], device=self.device)
        for count, window in enumerate(windows):
            result = F.conv2d(feature_dist_matrix.unsqueeze(1), window.unsqueeze(0).unsqueeze(0)).squeeze()  # B,K-3,K-3
            # 0 * inf = nan, so replace all nan
            result = torch.where(torch.isnan(result), torch.full_like(result, float("inf")), result)
            if count == 0:
                all_max = result.max(1)[0].max(1)[0]
            row_min, row_argmin = result.min(1)
            all_min, col_argmin = row_min.min(1)
            row_argmin = row_argmin[range(B), col_argmin]
            # min: all_min  coordinate: [row_argmin, col_argmin]
            all_min = torch.where(torch.isinf(all_min), all_max * 2, all_min)  # B
            dists += all_min
            # update search area
            mask = self.swin_masks[window.shape[0]][row_argmin, col_argmin]  # B,K,K
            feature_dist_matrix *= mask
        return dists

    def generate_swin_masks(self):
        """
        :return: dict{window_size: Tensor[K-window_size+1,K-window_size+1,K,K]}
        """
        ones = torch.ones([self.K, self.K], device=self.device)
        infs = torch.full([self.K, self.K], float("inf"), device=self.device)
        cols = torch.arange(self.K, device=self.device).expand(self.K, self.K)
        rows = cols.T
        swin_masks = {}
        for size in set(self.window_sizes):
            n = self.K - size + 1
            swin_masks[size] = torch.zeros([n, n, self.K, self.K], device=self.device)
            for i in range(n):
                left_top_row = i
                right_bottom_row = i + size - 1
                for j in range(n):
                    left_top_col = j
                    right_bottom_col = j + size - 1
                    bool_matrix = ((rows < left_top_row) & (cols < left_top_col)) | (
                        (rows > right_bottom_row) & (cols > right_bottom_col)
                    )
                    swin_masks[size][i, j] = torch.where(bool_matrix, ones, infs)
        return swin_masks


class SCF_OE(SCF_Module):
    def __init__(self, use_xyz=True):
        super().__init__("cube", 8, None)
        self.use_xyz = use_xyz
        input_channel = 6 if use_xyz else 3
        self.conv1 = nn.Sequential(
            self.conv_bn(input_channel, 3, [1, 2], [1, 2]),
            self.conv_bn(3, 3, [1, 2], [1, 2]),
            self.conv_bn(3, 3, [1, 2], [1, 2]),
        )
        self.conv2 = self.conv_bn(3, 3, [1, 1], [1, 1])

    def forward(self, points_xyz):
        """
        :param points_xyz: B,N,3
        :return: B,3,N
        """
        neighbor_idx = self.cube_neighbor(points_xyz, radius=self.cube_radius)  # B,N,8
        neighbor_xyz = self.gather_neighbor(points_xyz, neighbor_idx)  # B,N,8,3
        relative_dis, angle_alpha, angle_beta = self.local_polar_representation(points_xyz, neighbor_xyz)  # B,N,8
        feature = torch.stack([relative_dis, angle_alpha, angle_beta], dim=-1)  # B,N,8,3
        if self.use_xyz:
            feature = torch.cat([neighbor_xyz, feature], dim=-1)  # B,N,8,6
        feature = feature.permute(0, 3, 1, 2).contiguous()  # B,3|6,N,8
        feature_new = self.conv1(feature)  # B,3,N,1
        feature_new = self.conv2(feature_new)  # B,3,N,1
        feature_new = feature_new.squeeze(-1)  # B,3,N
        return feature_new

    @staticmethod
    def conv_bn(inp, oup, kernel, stride=1, activation="relu"):
        seq = nn.Sequential(nn.Conv2d(inp, oup, kernel, stride), nn.BatchNorm2d(oup))
        if activation == "relu":
            seq.add_module("2", nn.ReLU())
        return seq
