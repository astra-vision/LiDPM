import torch
import numpy as np
import open3d as o3d
from natsort import natsorted
import click
import tqdm
import MinkowskiEngine as ME
from datetime import datetime
import os
from os.path import join

from lidpm.utils.configs import save_config, smart_config
from lidpm.utils.pcd_preprocess import load_poses


# Main function
@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--data_path', '-in', type=str, help='path to the scan sequence')
@click.option('--gt_output_path', '-out', type=str, help='path to the scan sequence')
@click.option('--voxel_size', '-v', type=float, help='voxel size')
@click.option('--cpu', '-c', is_flag=True, help='Use CPU')
@click.option('--save_ply', '-ply', is_flag=True, help='Save additionally the .ply version of the GT maps')
def generate_gt(config_path, **kwargs):

    ctx = click.get_current_context()
    config = smart_config(config_path, ctx)

    device_label = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_label = 'cpu' if config['cpu'] else device_label
    device = torch.device(device_label)

    # save the updated configuration to the file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_config(config, join(config['gt_output_path'], f'config_{timestamp}.yaml'))

    for seq in tqdm.tqdm(['00','01','02','03','04','05','06','07','08','09','10']):
        print(f'Sequence {seq}.')
        torch.cuda.empty_cache()
        map_points = torch.empty((0,3)).to(device)
        if config['mode'] == 'labels':
            map_labels = torch.empty((0,), dtype=torch.int32).to(device)

        poses = load_poses(
            join(config['data_path'], seq),'calib.txt', 'poses.txt'
        )
        for pose, pcd_path in tqdm.tqdm(
                list(zip(poses, natsorted(os.listdir(join(config['data_path'], seq, 'velodyne')))))
        ):
            pose = torch.from_numpy(pose).float().to(device)
            pcd_file = join(config['data_path'], seq, 'velodyne', pcd_path)
            points = torch.from_numpy(np.fromfile(pcd_file, dtype=np.float32)).to(device)
            points = points.reshape(-1,4)

            label_file = pcd_file.replace('velodyne', 'labels').replace('.bin', '.label')
            l_set = torch.from_numpy(np.fromfile(label_file, dtype=np.uint32).astype(np.int32)).to(device)
            l_set_16 = l_set & 0xFFFF

            # remove moving points
            static_idx = (l_set_16 < 252) & (l_set_16 > 1)
            points = points[static_idx]

            # remove flying artifacts
            dist = torch.pow(points[:, :3], 2)
            dist = torch.sqrt(dist.sum(-1))
            points = points[dist > 3.5]

            points[:,-1] = 1.
            points = points @ pose.T

            map_points = torch.cat((map_points, points[:,:3]), axis=0)
            if config['mode'] == 'labels':
                map_labels = torch.cat((map_labels, l_set), axis=0)
            _, mapping = ME.utils.sparse_quantize(
                coordinates=map_points / config['voxel_size'], return_index=True, device=device_label
            )
            map_points = map_points[mapping]
            if config['mode'] == 'labels':
                map_labels = map_labels[mapping]


        print(f'saving map for sequence {seq}')
        os.makedirs(join(config['gt_output_path'], seq), exist_ok=True)
        save_to_path = join(
            config['gt_output_path'], seq,
            f'clean_map_voxelSize{str(config["voxel_size"]).replace(".", "")}.npy'
        )
        np.save(save_to_path, map_points.cpu().numpy())
        if config['mode'] == 'labels':
            np.save(save_to_path.replace('.npy', '_labels.npy'), map_labels.cpu().numpy())

        if config['save_ply']:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(map_points.cpu().numpy())
            o3d.io.write_point_cloud(save_to_path.replace('npy', 'ply'), pcd)


if __name__ == '__main__':
    generate_gt()
