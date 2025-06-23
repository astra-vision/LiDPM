import torch
import numpy as np
import open3d as o3d


def feats_to_coord(p_feats, resolution, bs, dim=3):
    p_feats = p_feats.reshape(bs, -1, dim)
    p_coord = torch.round(p_feats / resolution)

    return p_coord.reshape(-1, dim)


def point_set_to_sparse(p_full, p_part, n_full, n_part, filename, p_mean=None, p_std=None):
    concat_part = np.ceil(n_part / p_part.shape[0])
    p_part = p_part.repeat(concat_part, 0)
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_part)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)
    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    p_part = torch.tensor(np.array(pcd_part.points))

    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full))
    p_full = p_full[in_viewpoint]
    concat_full = np.ceil(n_full / p_full.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)

    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]




class SparseSegmentCollation:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
                'mean': torch.stack(batch[1]).float(),
                'std': torch.stack(batch[2]).float(),
                'pcd_part' if self.mode == 'diffusion' else 'pcd_noise': torch.stack(batch[3]).float(),
                'filename': batch[4],
                }
