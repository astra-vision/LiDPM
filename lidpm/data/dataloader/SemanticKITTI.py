from torch.utils.data import Dataset
from lidpm.utils.pcd_preprocess import load_poses
from lidpm.utils.pcd_transforms import *
from lidpm.utils.collations import point_set_to_sparse
from natsort import natsorted
import os
import numpy as np

from glob import glob

import warnings
from os.path import join

warnings.filterwarnings('ignore')


#################################################
################## Data loader ##################
#################################################

class KITTIDataSet(Dataset):
    def __init__(self, data_dir, gt_map_dir, seqs, split, resolution, num_points, range_limits, duplication_factor,
                 minival_path, num_validation_pointclouds):
        super().__init__()
        self.data_dir = data_dir
        self.gt_map_dir = gt_map_dir
        self.minival_path = minival_path

        self.n_clusters = 50
        self.resolution = resolution
        self.num_points = num_points
        self.range_limits = range_limits

        self.split = split
        self.seqs = seqs
        self.cache_maps = {}
        self.duplication_factor = duplication_factor
        self.seed = 42
        self.validation_size = num_validation_pointclouds

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list(self.split)
        self.data_stats = {'mean': None, 'std': None}

        self.nr_data = len(self.points_datapath)
        print('\n The size of %s data is %d.\n' % (self.split, len(self.points_datapath)))

    def datapath_list(self, mode):
        self.points_datapath = []
        self.seq_poses = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            poses = load_poses(point_seq_path, 'calib.txt', 'poses.txt')
            p_full = np.load(
                f'{self.gt_map_dir}/{seq}/clean_map_voxelSize01.npy') if self.split != 'test' else np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.cache_maps[seq] = p_full

            if mode == 'train':
                point_seq_bin = natsorted(glob(join(point_seq_path, 'velodyne', '*.bin')))
            elif mode == 'validation':
                with open(self.minival_path, 'r') as file:
                    point_seq_bin = [line.strip() for line in file][: self.validation_size]

            for file_name in point_seq_bin:
                idx = int(os.path.basename(file_name).split('.')[0])
                self.points_datapath.append(file_name)
                self.seq_poses.append(poses[idx])

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:, :, :3] = rotate_point_cloud(points[:, :, :3])
        points[:, :, :3] = rotate_perturbation_point_cloud(points[:, :, :3])
        points[:, :, :3] = random_scale_point_cloud(points[:, :, :3])
        points[:, :, :3] = random_flip_point_cloud(points[:, :, :3])

        return np.squeeze(points, axis=0)

    def clamp_in_range(self, pcd, pose=np.eye(4), vehicle_radius=-1):
        trans = pose[:-1, -1]
        if 'max_range' in self.range_limits:
            dist_part = np.sum((pcd - trans) ** 2, -1) ** .5
            pcd = pcd[(dist_part < self.range_limits['max_range']) & (dist_part > vehicle_radius)]
            pcd = np.concatenate((pcd, np.ones((len(pcd), 1))), axis=-1)
            pcd = (pcd @ np.linalg.inv(pose).T)[:, :3]
            pcd = pcd[pcd[:, 2] > -4.]
        elif 'min_limits' in self.range_limits:
            pcd = np.concatenate((pcd, np.ones((len(pcd), 1))), axis=-1)
            pcd = (pcd @ np.linalg.inv(pose).T)[:, :3]
            min_margins = np.array(self.range_limits['min_limits']) - self.range_limits['margin']
            max_margins = np.array(self.range_limits['max_limits']) + self.range_limits['margin']
            in_box_mask = (
                    (pcd[:, 0] >= min_margins[0]) & (pcd[:, 0] <= max_margins[0]) &
                    (pcd[:, 1] >= min_margins[1]) & (pcd[:, 1] <= max_margins[1]) &
                    (pcd[:, 2] >= min_margins[2]) & (pcd[:, 2] <= max_margins[2])
            )
            pcd = pcd[in_box_mask]
            dist_part = np.sum(pcd ** 2, -1) ** .5
            pcd = pcd[dist_part > vehicle_radius]

        else:
            raise ValueError('Unrecognized type of clamping range limits')
        return pcd

    def __getitem__(self, index):
        seq_num = self.points_datapath[index].split('/')[-3]

        p_part = np.fromfile(self.points_datapath[index], dtype=np.float32)
        p_part = p_part.reshape((-1, 4))[:, :3]

        if self.split != 'test':
            label_file = self.points_datapath[index].replace('velodyne', 'labels').replace('.bin', '.label')
            l_set = np.fromfile(label_file, dtype=np.uint32)
            l_set = l_set.reshape((-1))
            l_set = l_set & 0xFFFF
            static_idx = (l_set < 252) & (l_set > 1)
            p_part = p_part[static_idx]

        p_part = self.clamp_in_range(p_part, vehicle_radius=3.5)
        pose = self.seq_poses[index]

        p_map = self.cache_maps[seq_num]

        if self.split != 'test':
            p_full = self.clamp_in_range(p_map, pose=pose, vehicle_radius=0.)
        else:
            p_full = p_part

        if self.split == 'train':
            p_concat = np.concatenate((p_full, p_part), axis=0)
            p_concat = self.transforms(p_concat)

            p_full = p_concat[:-len(p_part)]
            p_part = p_concat[-len(p_part):]

        # partial pcd has 1/10 of the complete pcd size
        n_part = int(self.num_points / self.duplication_factor)

        return point_set_to_sparse(
            p_full,  # gt
            p_part,  # initial
            self.num_points,
            n_part,
            self.points_datapath[index],
            p_mean=self.data_stats['mean'],
            p_std=self.data_stats['std'],
        )

    def __len__(self):
        return self.nr_data

##################################################################################################
