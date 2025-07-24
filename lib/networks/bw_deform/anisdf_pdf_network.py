import torch.nn as nn
#import spconv
import torch.nn.functional as F
import torch
from pyexpat import features
import math

from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
import os
from lib.utils import sample_utils
from lib.networks.mip import *
import spconv.pytorch as spconv

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.resd_latent = nn.Embedding(cfg.num_latent_code, 128)

        self.actvn = nn.ReLU()

        input_ch = 168 #168 IPE 135 PE
        D = 8
        W = 256
        self.skips = [4]
        self.resd_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.resd_fc = nn.Conv1d(W, 3, 1)
        self.resd_fc.bias.data.fill_(0)

        if cfg.get('init_sdf', False):
            init_path = os.path.join('data/trained_model', cfg.task,
                                     cfg.init_sdf)
            net_utils.load_network(self,
                                   init_path,
                                   only=['tpose_human.sdf_network'])

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_residual_deformation(self, tpose, batch, covs):

        # add
        light_pts = integrated_pos_enc(
            tpose,
            covs,
            0,
            16,
        )
        light_pts_size = light_pts.shape[0] * light_pts.shape[1]
        pts = light_pts.unsqueeze(0).reshape(1, light_pts_size, -1)

        # pts = embedder.xyz_embedder(tpose)

        pts = pts.transpose(1, 2)
        latent = batch['poses']
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)

        net = features
        for i, l in enumerate(self.resd_linears):
            net = self.actvn(self.resd_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        resd = self.resd_fc(net)
        resd = resd.transpose(1, 2)
        resd = 0.05 * torch.tanh(resd)
        return resd

    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch, covs):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        pbw = pbw.permute(0, 2, 1) #(1,left_n_samples,24)剩余点的混合权重

        # transform points from i to i_0
        init_tpose = pose_points_to_tpose_points(pose_pts, pbw,
                                                 batch['A']) #采样点转到正则空间即tpose下的初始坐标
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw,
                                                   batch['big_A'])

        resd = self.calculate_residual_deformation(init_bigpose, batch, covs)
        tpose = init_bigpose + resd #resd为非刚性部分的位移场

        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pbw,
                                                 batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pbw,
                                                 batch['big_A'])
        else:
            tpose_dirs = None

        return tpose, tpose_dirs, init_bigpose, resd

    def calculate_bigpose_smpl_bw(self, bigpose, input_bw):
        smpl_bw = pts_sample_blend_weights(bigpose, input_bw['tbw'],
                                           input_bw['tbounds'])
        return smpl_bw

    def calculate_wpts_sdf(self, wpts, batch, covs):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        # transform points from the pose space to the tpose space
        tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
            pose_pts, None, batch, covs)
        tpose = tpose[0]
        sdf = self.tpose_human.sdf_network(tpose, batch)[:, :1]

        return sdf

    def wpts_gradient(self, wpts, batch):
        wpts.requires_grad_(True)
        with torch.enable_grad():
            sdf = self.calculate_wpts_sdf(wpts, batch)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients

    def gradient_of_deformed_sdf(self, x, batch, covs):
        x.requires_grad_(True)
        with torch.enable_grad():
            resd = self.calculate_residual_deformation(x, batch, covs)
            tpose = x + resd
            tpose = tpose[0]
            y = self.tpose_human.sdf_network(tpose, batch, covs)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients, y[None]

    def forward(self, wpts, covs, viewdir, dists, batch, means_covs):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        covs = covs[None]
        viewdir = viewdir[None]
        pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])

        #add
        # device = wpts.device
        # directions = [
        #     torch.tensor([-1, 1, 1], device=device),  # (-x, y, z)
        #     torch.tensor([1, -1, 1], device=device),  # (x, -y, z)
        #     torch.tensor([1, 1, -1], device=device),  # (x, y, -z)
        #     torch.tensor([-1, 1, -1], device=device),  # (-x, y, -z)
        #     torch.tensor([-1, -1, 1], device=device),  # (-x, -y, z)
        #     torch.tensor([1, -1, -1], device=device),  # (x, -y, -z)
        #     torch.tensor([-1, -1, -1], device=device),  # (-x, -y, -z)
        #     torch.tensor([1, 1, 1], device=device)  # (x, y, z)
        # ]
        # # covs[..., 2] = torch.clamp(covs[..., 2], max=4e-6)
        # covs_sqrt = torch.sqrt(covs)
        # transformed_tensors = [covs_sqrt * direction for direction in directions]
        # final_tensors = [pose_pts * (1 + transformed) for transformed in transformed_tensors]
        # final_tensor = torch.cat([pose_pts] + final_tensors, dim=0)
        # viewdir = viewdir.repeat(9, 1, 1).contiguous()
        # pose_dirs = pose_dirs.repeat(9, 1, 1).contiguous()
        # pose_pts = final_tensor.view(wpts.shape[0], -1, wpts.shape[-1])
        # viewdir = viewdir.view(wpts.shape[0], -1, wpts.shape[-1])
        # pose_dirs = pose_dirs.view(wpts.shape[0], -1, wpts.shape[-1])

        with torch.no_grad():
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights']) #输入姿态空间下采样点和顶点，关节点对顶点影响的权重；输出最近五个顶点的加权影响权重作为采样点权重pbw
            pnorm = pnorm[..., 0] #(1,65536) 最近五个点的加权距离
            norm_th = 0.1 #通过阈值过滤了大部分距离顶点较远的采样点
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]
            viewdir = viewdir[pind][None]
            pose_dirs = pose_dirs[pind][None]
            covs = covs[pind][None]

        # transform points from the pose space to the tpose space
        tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
            pose_pts, pose_dirs, batch, covs)
        tpose = tpose[0]
        if cfg.tpose_viewdir:
            viewdir = tpose_dirs[0]
        else:
            viewdir = viewdir[0]
        ret = self.tpose_human(tpose, viewdir, dists, batch, covs, pose_pts[0])

        ind = ret['sdf'][:, 0].detach().abs() < 0.02
        init_bigpose = init_bigpose[0][ind][None].detach().clone()
        covs = covs[0][ind][None].detach().clone()

        if ret['raw'].requires_grad and ind.sum() != 0:
            observed_gradients, _ = self.gradient_of_deformed_sdf(
                init_bigpose, batch, covs)
            ret.update({'observed_gradients': observed_gradients})

        tbounds = batch['tbounds'][0]
        tbounds[0] -= 0.05
        tbounds[1] += 0.05
        inside = tpose > tbounds[:1]
        inside = inside * (tpose < tbounds[1:])
        outside = torch.sum(inside, dim=1) != 3
        ret['raw'][outside] = 0

        n_batch, n_point = wpts.shape[:2]
        raw = torch.zeros([n_batch, n_point, 4]).to(wpts)
        raw[pind] = ret['raw']
        sdf = 10 * torch.ones([n_batch, n_point, 1]).to(wpts)
        sdf[pind] = ret['sdf']
        ret.update({'raw': raw, 'sdf': sdf, 'resd': resd})
        if 'gradients' in ret:
            ret.update({'gradients': ret['gradients'][None]})

        return ret

    def get_sdf(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        with torch.no_grad():
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
            pnorm = pnorm[..., 0]
            norm_th = 0.1
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]

        # initial blend weights of points at i
        pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        pbw = pbw.permute(0, 2, 1)

        # transform points from i to i_0
        init_tpose = pose_points_to_tpose_points(pose_pts, pbw,
                                                 batch['A'])
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw,
                                                   batch['big_A'])
        resd = self.calculate_residual_deformation(init_bigpose, batch)
        tpose = init_bigpose + resd
        tpose = tpose[0]

        sdf_nn_output = self.tpose_human.sdf_network(tpose, batch)
        sdf = sdf_nn_output[:, 0]

        n_batch, n_point = wpts.shape[:2]
        sdf_full = 10 * torch.ones([n_batch, n_point]).to(wpts)
        sdf_full[pind] = sdf
        sdf = sdf_full.view(-1, 1)

        return sdf


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.sdf_network = SDFNetwork()
        self.beta_network = BetaNetwork()
        self.color_network = ColorNetwork()
        self.latent_code_net = LatentNetwork()

    def sdf_to_alpha(self, sdf, beta):
        x = -sdf

        # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
        ind0 = x <= 0
        val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

        # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
        ind1 = x > 0
        val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

        val = torch.zeros_like(sdf)
        val[ind0] = val0
        val[ind1] = val1

        return val

    def forward(self, wpts, viewdir, dists, batch, covs, pose_pts):
        # calculate sdf
        wpts.requires_grad_()
        with torch.enable_grad():
            sdf_nn_output = self.sdf_network(wpts, batch, covs)
            # sdf_nn_output = self.latent_code_net(wpts, batch)
            sdf = sdf_nn_output[:, :1]

        feature_vector = sdf_nn_output[:, 1:]

        # calculate normal
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        # gradients = self.sdf_network.gradient(wpts, batch)[:, 0]
        #add
        # feature_vector = self.latent_code_net(wpts, batch, feature_vector, covs) #tpose
        feature_vector = self.latent_code_net(pose_pts, batch, feature_vector, covs) #ppose

        # calculate alpha
        wpts = wpts.detach()
        beta = self.beta_network(wpts).clamp(1e-9, 1e6)
        alpha = self.sdf_to_alpha(sdf, beta)
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * 0.005)
        alpha = raw2alpha(alpha[:, 0], dists)

        # calculate color
        ind = batch['latent_index']
        rgb = self.color_network(wpts, gradients, viewdir, feature_vector, ind)
        # rgb = self.color_network(wpts, viewdir, feature_vector, ind)

        raw = torch.cat((rgb, alpha[:, None]), dim=1)
        ret = {'raw': raw, 'sdf': sdf, 'gradients': gradients}
        # ret = {'raw': raw, 'sdf': sdf}

        return ret

###add###
class GSU(nn.Module):
    def __init__(self, hidden_dim):
        super(GSU, self).__init__()
        self.hidden_dim = hidden_dim
        # 用于计算门控的权重
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, task_j_features, task_k_features):
        # 计算门控值 (g^l_jk)
        task_k_features = task_k_features.transpose(1,2)
        gate_value = torch.sigmoid(self.gate(task_k_features))
        # 通过门控机制选择有用的特征
        selected_features = gate_value * task_k_features
        # 返回融合后的特征
        return task_j_features + selected_features.transpose(1,2)

class LatentNetwork(nn.Module):
    def __init__(self):
        super(LatentNetwork, self).__init__()
        self.c = nn.Embedding(6890, 16)  # 潜在代码 z 的维度设置为 16
        self.xyzc_net = SparseConvNet()
        self.fc_means = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.gsu = GSU(352)
        # self.lin_la = nn.Conv1d(in_channels=352, out_channels=256, kernel_size=3, stride=1, padding=1)

        d_in = 608
        d_out = 256
        d_hidden = 256
        n_layers = 2

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        multires = 1

        skip_in = []
        bias = 0.5
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                               np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def encode_sparse_voxels(self, sp_input):
        coord_vertices = sp_input['coord_vertices']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        code = self.c(torch.arange(0, 6890).to(coord_vertices.device))
        xyzc = spconv.SparseConvTensor(code, coord_vertices, out_sh, batch_size)
        feature_volume = self.xyzc_net(xyzc) #通过卷积和下采样，输入代码扩散到附近的空间。稀疏卷积操作

        return feature_volume

    def prepare_sp_input(self, batch):
        sp_input = {}
        sh = batch['coord_vertices'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord_vertices'])
        coord = batch['coord_vertices'].view(-1, sh[-1])
        sp_input['coord_vertices'] = torch.cat([idx[:, None], coord], dim=1)
        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]
        sp_input['ptbounds'] = batch['ptbounds']
        sp_input['R'] = batch['R']
        sp_input['Th'] = batch['Th']
        return sp_input

    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)  # 两个张量矩阵相乘
        return pts

    def get_grid_coords(self, pts, sp_input):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]  # 把所有的三维坐标进行了一个【2，1，0】三个位置的转变
        min_dhw = sp_input['ptbounds'][:, 0, [2, 1, 0]]  # 把bounds里第一行三位数组做了变换
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # max_values = torch.max(dhw)
        # min_values = torch.min(dhw)
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def interpolate_features(self, grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,
                                    grid_coords,
                                    padding_mode='zeros',
                                    align_corners=True)
            features.append(feature)
            # max_values = torch.max(volume)
            # min_values = torch.min(volume)
            # max_values = torch.max(feature)
            # min_values = torch.min(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        return features

    def generate_spherical_directions(self,num_points, device='cuda'):
        """Generate num_points directions evenly distributed on a sphere."""
        directions = []
        phi = math.pi * (3. - math.sqrt(5.))  # Golden angle in radians
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at a given y
            theta = phi * i  # Golden angle
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            directions.append(torch.tensor([x, y, z], device=device))
        return directions

    def forward(self, inputs, batch, feature, covs):
        sp_input = self.prepare_sp_input(batch)
        feature_volume = self.encode_sparse_voxels(sp_input)

        # add
        inputs = inputs[None]
        device = inputs.device
        directions = [
            torch.tensor([-1, 1, 1], device=device),  # (-x, y, z)
            torch.tensor([1, -1, 1], device=device),  # (x, -y, z)
            torch.tensor([1, 1, -1], device=device),  # (x, y, -z)
            torch.tensor([-1, 1, -1], device=device),  # (-x, y, -z)
            torch.tensor([-1, -1, 1], device=device),  # (-x, -y, z)
            torch.tensor([1, -1, -1], device=device),  # (x, -y, -z)
            torch.tensor([-1, -1, -1], device=device),  # (-x, -y, -z)
            torch.tensor([1, 1, 1], device=device)  # (x, y, z)
        ]

        #球坐标系方向向量
        # directions = self.generate_spherical_directions(8, device='cuda')

        covs_sqrt = torch.sqrt(covs)
        transformed_tensors = [covs_sqrt * direction for direction in directions]
        final_tensors = [inputs * (1 + transformed) for transformed in transformed_tensors]
        final_tensor = torch.cat([inputs] + final_tensors, dim=0)
        xyzc_features_list = []
        for tensor in final_tensor:
            # ppts = self.pts_to_can_pts(tensor, sp_input)  # transform pts from the world coordinate to the smpl coordinate
            ppts = tensor
            grid_coords = self.get_grid_coords(ppts, sp_input)  # convert xyz to the voxel coordinate dhw
            grid_coords = grid_coords[:, None, None]
            xyzc_features = self.interpolate_features(grid_coords, feature_volume)
            # max_values = torch.max(xyzc_features)
            # min_values = torch.min(xyzc_features)
            xyzc_features_list.append(xyzc_features)
        res_features = xyzc_features_list[0]
        origin_xyzc_feature_cat = torch.cat(xyzc_features_list[1:], dim=0)
        origin_xyzc_feature_cat = origin_xyzc_feature_cat.view(1, 8, -1)  # (1,9,352*65536)
        xyzc_features = self.fc_means(origin_xyzc_feature_cat).view(1, 352, -1)
        xyzc_features = self.gsu(res_features, xyzc_features)


        # # ppts = self.pts_to_can_pts(inputs, sp_input)
        # ppts = inputs
        # grid_coords = self.get_grid_coords(ppts, sp_input)  # convert xyz to the voxel coordinate dhw
        # grid_coords = grid_coords[:, None, None]
        # xyzc_features = self.interpolate_features(grid_coords, feature_volume)   #(1，352，left_points)


        x = xyzc_features.transpose(1, 2).squeeze(0)
        x = torch.cat([x, feature], dim=1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, xyzc_features], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(16, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):#6890 -》 人体
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
###


class SDFNetwork(nn.Module):
    def __init__(self):
        super(SDFNetwork, self).__init__()

        d_in = 3
        d_out = 257
        d_hidden = 256
        n_layers = 8

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
            #add
            # dims[0] = 48

        skip_in = [4]
        bias = 0.5
        scale = 1
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, batch, covs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
            # add
            # light_pts = integrated_pos_enc(
            #     inputs.unsqueeze(0),
            #     covs,
            #     0,
            #     8,
            # )
            # light_pts_size = light_pts.shape[0] * light_pts.shape[1]
            # inputs = light_pts.unsqueeze(0).reshape(light_pts_size, -1)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, batch):
        return self.forward(x, batch)[:, :1]

    def gradient(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x, batch)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class BetaNetwork(nn.Module):
    def __init__(self):
        super(BetaNetwork, self).__init__()
        init_val = 0.1
        self.register_parameter('beta', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        beta = self.beta
        # beta = torch.exp(self.beta).to(x)
        return beta


class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        self.color_latent = nn.Embedding(cfg.num_latent_code, 128)

        d_feature = 256
        mode = 'idr'
        #mode = 'no_normal'
        d_in = 9
        d_out = 3
        d_hidden = 256
        n_layers = 4
        squeeze_out = True

        if not cfg.get('color_with_viewdir', True): #不存在默认true
            mode = 'no_view_dir'

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden
                                     for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if self.mode != 'no_view_dir':
            multires_view = 4
            if multires_view > 0:
                embedview_fn, input_ch = embedder.get_embedder(multires_view)
                self.embedview_fn = embedview_fn
                dims[0] += (input_ch - 3)
        else:
            dims[0] = dims[0] - 3

        if self.mode == 'no_normal':
            dims[0] = dims[0] - 3

        self.num_layers = len(dims)

        self.lin0 = nn.Linear(dims[0], d_hidden)
        self.lin1 = nn.Linear(d_hidden, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden)
        self.lin3 = nn.Linear(d_hidden + 128, d_hidden)
        self.lin4 = nn.Linear(d_hidden, d_out)

        weight_norm = True
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors,
                latent_index):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat(
                [points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors],
                                        dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors],
                                        dim=-1)

        x = rendering_input

        net = self.relu(self.lin0(x))
        net = self.relu(self.lin1(net))
        net = self.relu(self.lin2(net)) #(n_samples, 256)

        latent = self.color_latent(latent_index)
        latent = latent.expand(net.size(0), latent.size(1))
        features = torch.cat((net, latent), dim=1)

        net = self.relu(self.lin3(features))
        x = self.lin4(net)

        if self.squeeze_out:
            x = torch.sigmoid(x)

        return x
