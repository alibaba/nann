"""
    Class Boundary Detection Loss
    Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations
    https://github.com/jiwoon-ahn/irn/blob/master/step/train_irn.py
"""
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import numpy as np
import random
sys.path.append(os.path.dirname(__file__))

__all__ = ['CBDLoss',
        ]

class PathIndex:

    def __init__(self, radius=5, default_size=None):

        self.radius = radius
        self.radius_floor = int(np.ceil(radius) - 1)

        self.path_list_by_length, self.path_dst = self.get_all_dir_paths(self.radius)
        self.path_dst2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.path_dst).transpose(1, 0), 0), -1).float()

        if default_size:
            self.default_path_indices, self.default_src_indices, self.default_dst_indices \
                = self.get_path_indices(default_size)

        return

    def get_all_dir_paths(self, max_radius=5):

        coord_indices_by_length = [[] for _ in range(max_radius * 4)]

        search_dirs = []

        for x in range(1, max_radius):
            search_dirs.append((0, x))

        for y in range(1, max_radius):
            for x in range(-max_radius + 1, max_radius):
                if x * x + y * y < max_radius ** 2:
                    search_dirs.append((y, x))

        for dir in search_dirs:

            length_sq = dir[0] ** 2 + dir[1] ** 2
            path_coords = []

            min_y, max_y = sorted((0, dir[0]))
            min_x, max_x = sorted((0, dir[1]))

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):

                    dist_sq = (dir[0] * x - dir[1] * y) ** 2 / length_sq

                    if dist_sq < 1:
                        path_coords.append([y, x])

            path_coords.sort(key=lambda x: -abs(x[0]) - abs(x[1]))
            path_length = len(path_coords)

            coord_indices_by_length[path_length].append(path_coords)

        path_list_by_length = [np.asarray(v) for v in coord_indices_by_length if v]
        path_destinations = np.concatenate([p[:, 0] for p in path_list_by_length], axis=0)

        return path_list_by_length, path_destinations

    def get_path_indices(self, size):

        full_indices = np.reshape(np.arange(0, size[0] * size[1], dtype=np.int64), (size[0], size[1]))

        cropped_height = size[0] - self.radius_floor
        cropped_width = size[1] - 2 * self.radius_floor

        paths_by_length_list = []

        for paths in self.path_list_by_length:

            path_indices_list = []
            for p in paths:

                coord_indices_list = []

                for dy, dx in p:
                    coord_indices = full_indices[dy:dy + cropped_height,
                                    self.radius_floor + dx:self.radius_floor + dx + cropped_width]
                    coord_indices = np.reshape(coord_indices, [-1])

                    coord_indices_list.append(coord_indices)

                path_indices_list.append(coord_indices_list)

            paths_by_length_list.append(np.array(path_indices_list))

        src_indices = np.reshape(full_indices[:cropped_height, self.radius_floor:self.radius_floor + cropped_width], -1)
        dest_indices = np.concatenate([p[:,0] for p in paths_by_length_list], axis=0)

        return paths_by_length_list, \
               src_indices, \
               dest_indices

    def to_displacement(self, x):
        height, width = x.size(2), x.size(3)

        cropped_height = height - self.radius_floor
        cropped_width = width - 2 * self.radius_floor

        feat_src = x[:, :, :cropped_height, self.radius_floor:self.radius_floor + cropped_width]

        feat_dest = [x[:, :, dy:dy + cropped_height, self.radius_floor + dx:self.radius_floor + dx + cropped_width]
                       for dy, dx in self.path_dst]
        feat_dest = torch.stack(feat_dest, 2)

        disp = torch.unsqueeze(feat_src, 2) - feat_dest
        disp = disp.view(disp.size(0), disp.size(1), disp.size(2), -1)

        return disp

    def to_displacement_loss(self, x):

        return torch.abs(x - self.path_dst2.cuda())


class AffinityDisplacement(nn.Module):

    path_indices_prefix = "path_indices"

    def __init__(self, path_indices=None, ind_from=None, ind_to=None):

        super(AffinityDisplacement, self).__init__()

        self.n_path_type = len(path_indices)
        for i in range(self.n_path_type):
            param = torch.nn.Parameter(torch.from_numpy(path_indices[i]), requires_grad=False)
            self.register_parameter(AffinityDisplacement.path_indices_prefix + str(i), param)

        # self.register_parameter(
        #     "ind_from",
        #     torch.nn.Parameter(torch.unsqueeze(ind_from, dim=0), requires_grad=False))
        # self.register_parameter(
        #     "ind_to",
        #     torch.nn.Parameter(ind_to, requires_grad=False))


    def edge_to_affinity(self, edge):

        aff_list = []
        edge = edge.view(edge.size(0), -1)

        for i in range(self.n_path_type):
            ind = self._parameters[AffinityDisplacement.path_indices_prefix + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)

        return aff_cat


    def forward(self, x):
        aff_out = self.edge_to_affinity(x)

        return aff_out


class CBDLoss(nn.Module):
    def __init__(self, crop_size, radius=10, ignore=255):
        super(CBDLoss, self).__init__()
        self.ignore = ignore
        path_index = PathIndex(
            radius=radius, default_size=(crop_size, crop_size))
        self.affinity_displacement = AffinityDisplacement(path_index.default_path_indices)

    def forward(self, pred_output, pos_label, neg_label):
        edge = self.affinity_displacement(pred_output)

        pos_aff_loss = torch.sum(
            - pos_label * torch.log(edge + 1e-5)) / (torch.sum(pos_label) + 1e-5)
        neg_aff_loss = torch.sum(
            - neg_label * torch.log(1. + 1e-5 - edge)) / (torch.sum(neg_label) + 1e-5)
        return (pos_aff_loss + neg_aff_loss)/2