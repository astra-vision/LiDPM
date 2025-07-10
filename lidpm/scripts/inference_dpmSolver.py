import numpy as np
import MinkowskiEngine as ME
import torch
import lidpm.models.minkunet as minknet
import open3d as o3d
from diffusers import DPMSolverMultistepScheduler
from pytorch_lightning.core.lightning import LightningModule
import yaml
import os
import tqdm
from glob import glob
import click
import time
from os.path import join
from lidpm.utils.configs import smart_config, save_config
from datetime import datetime

from pytorch_lightning import seed_everything


class DiffCompletion(LightningModule):
    def __init__(self, config, full_exp_dir):
        super().__init__()

        ckpt_diff = torch.load(config['diff'])
        denoising_steps = config['denoising_steps']
        self.starting_point = config['starting_point']
        self.save_hyperparameters(ckpt_diff['hyper_parameters'])
        assert denoising_steps <= self.hparams['diff']['t_steps'], \
            f"The number of denoising steps on inference cannot be bigger than it was trained with, T={self.hparams['diff']['t_steps']} (you've set '-T {denoising_steps}')"

        self.partial_enc = minknet.MinkGlobalEncIN(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.model = minknet.MinkUNetDiffIN(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.load_state_dict(ckpt_diff['state_dict'], strict=False)

        self.partial_enc.eval()
        self.model.eval()
        self.cuda()

        full_betas = self.linear_beta_schedule(
            self.hparams['diff']['t_steps'],
            self.hparams['diff']['beta_start'],
            self.hparams['diff']['beta_end'],
        )
        beta_end = full_betas[self.starting_point].item()

        self.betas = self.linear_beta_schedule(
            self.starting_point,
            self.hparams['diff']['beta_start'],
            beta_end
        )

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=self.device
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)

        # Not to overwrite the self.hparams from the checkpoints, but to create a separate field for the inference values
        self.hparams['inference'] = {}
        self.hparams['inference']['s_steps'] = denoising_steps

        self.dpm_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.starting_point,
            beta_start=self.hparams['diff']['beta_start'],
            beta_end=beta_end,
            beta_schedule='linear',
            algorithm_type='sde-dpmsolver++',
            solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(denoising_steps)
        self.scheduler_to_cuda()

        self.hparams['inference']['uncond_w'] = config['cond_weight']
        self.w_uncond = self.hparams['inference']['uncond_w']

        range_limits = {
            "ssc_kitti_box": {
                "min_limits": [0, -25.6, -2],
                "max_limits": [51.2, 25.6, 4.4],
                "margin": 0.2
            },
            "radius": {
                "max_range": 50
            }
        }
        self.range_limits = range_limits[config['range']]
        self.hparams['inference']['range'] = config['range']

        self.hparams['inference']['num_points'] = config['num_points']
        self.hparams['inference']['duplication_factor'] = config['duplication_factor']

        self.lidar_height = 1.73

        with open(join(full_exp_dir, 'exp_config.yaml'), 'w+') as exp_config:
            yaml.dump(self.hparams, exp_config)

    @staticmethod
    def linear_beta_schedule(timesteps, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, timesteps)

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def points_to_tensor(self, points):
        x_feats = ME.utils.batched_coordinates(list(points[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord = torch.round(x_coord / self.hparams['data']['resolution'])

        x_t = ME.TensorField(
            features=x_feats[:, 1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def reset_partial_pcd(self, x_part):
        x_part = self.points_to_tensor(x_part.F.reshape(1, -1, 3).detach())
        x_uncond = self.points_to_tensor(torch.zeros_like(x_part.F.reshape(1, -1, 3)))

        return x_part, x_uncond

    def add_vehicle_disc(self, pcd, num_points=1000, radius=3.5):
        # Generate random points
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = np.sqrt(np.random.uniform(0, radius ** 2, num_points))

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = - self.lidar_height * np.ones_like(x)

        return np.concatenate((np.column_stack((x, y, z)), pcd), axis=0)

    def preprocess_scan(self, scan):
        if 'max_range' in self.range_limits.keys():
            dist = np.sqrt(np.sum((scan) ** 2, axis=-1))
            scan = scan[(dist < self.range_limits['max_range']) & (dist > 3.5)][:, :3]
        else:
            min_margins = np.array(self.range_limits['min_limits']) - self.range_limits['margin']
            max_margins = np.array(self.range_limits['max_limits']) + self.range_limits['margin']
            in_box_mask = (
                    (scan[:, 0] >= min_margins[0]) & (scan[:, 0] <= max_margins[0]) &
                    (scan[:, 1] >= min_margins[1]) & (scan[:, 1] <= max_margins[1]) &
                    (scan[:, 2] >= min_margins[2]) & (scan[:, 2] <= max_margins[2])
            )
            scan = scan[in_box_mask]
            dist_part = np.sum(scan ** 2, -1) ** .5
            scan = scan[dist_part > 3.5]

        input_pcd = scan.copy()

        # part with conditioning
        cond_pcd_scan = o3d.geometry.PointCloud()
        cond_pcd_scan.points = o3d.utility.Vector3dVector(scan)
        cond_pcd_scan = cond_pcd_scan.farthest_point_down_sample(
            int(self.hparams['inference']['num_points'] / self.hparams['inference']['duplication_factor'])
        )
        cond_pcd = torch.tensor(np.array(cond_pcd_scan.points))[None].to(self.device)

        input_pcd = self.add_vehicle_disc(input_pcd)

        input_pcd_scan = o3d.geometry.PointCloud()
        input_pcd_scan.points = o3d.utility.Vector3dVector(input_pcd)
        input_pcd_scan = input_pcd_scan.farthest_point_down_sample(
            int(self.hparams['inference']['num_points'] / self.hparams['inference']['duplication_factor'])
        )
        input_pcd = torch.tensor(np.array(input_pcd_scan.points))[None].to(self.device)
        input_pcd = input_pcd.repeat(1, self.hparams['inference']['duplication_factor'], 1)
        return input_pcd, cond_pcd

    def postprocess_scan(self, completed_scan, input_scan):
        if 'max_range' in self.range_limits.keys():
            dist = np.sqrt(np.sum((completed_scan) ** 2, axis=-1))
            post_scan = completed_scan[(dist < self.range_limits['max_range'])]
        else:
            min_margins = np.array(self.range_limits['min_limits']) - self.range_limits['margin']
            max_margins = np.array(self.range_limits['max_limits']) + self.range_limits['margin']
            in_box_mask = (
                    (completed_scan[:, 0] >= min_margins[0]) & (completed_scan[:, 0] <= max_margins[0]) &
                    (completed_scan[:, 1] >= min_margins[1]) & (completed_scan[:, 1] <= max_margins[1]) &
                    (completed_scan[:, 2] >= min_margins[2]) & (completed_scan[:, 2] <= max_margins[2])
            )
            post_scan = completed_scan[in_box_mask]

        max_z = input_scan[..., 2].max().item()
        min_z = (input_scan[..., 2].mean() - 2 * input_scan[..., 2].std()).item()

        post_scan = post_scan[(post_scan[:, 2] < max_z) & (post_scan[:, 2] > min_z)]

        return post_scan

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:, None, None].to(self.device) * x + \
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None].to(self.device) * noise

    def denoise_scan(self, scan):
        input_pcd, cond_pcd = self.preprocess_scan(scan)

        if self.starting_point > 0:
            noise = torch.randn(input_pcd.shape, device=self.device)
            t = torch.ones(1).long().to(self.device) * (self.starting_point - 1)
            q_sampled_noised_pcd = self.q_sample(input_pcd, t, noise)

            x_full = self.points_to_tensor(q_sampled_noised_pcd)
            x_cond = self.points_to_tensor(cond_pcd)
            x_uncond = self.points_to_tensor(torch.zeros_like(cond_pcd))

            completed_scan = self.completion_loop(x_full, x_cond, x_uncond)
            post_scan = self.postprocess_scan(completed_scan, cond_pcd)
        else:
            post_scan = input_pcd

        return post_scan

    def forward(self, x_full, x_full_sparse, x_part, t):
        with torch.no_grad():
            part_feat = self.partial_enc(x_part)
            out = self.model(x_full, x_full_sparse, part_feat, t)

        torch.cuda.empty_cache()
        return out.reshape(t.shape[0], -1, 3)

    def completion_loop(self, x_full, x_cond, x_uncond):
        # x_init - clean duplicated pointcloud
        # x_t - noised TensorField pointcloud
        # x_cond - clean TensorField conditioning pointcloud

        for t in tqdm.tqdm(self.dpm_scheduler.timesteps):
            t_tensor = torch.tensor([t], device=self.device)
            x_full_sparse = x_full.sparse()

            est_noise_cond = self.forward(x_full, x_full_sparse, x_cond, t_tensor)
            est_noise_zero_cond = self.forward(x_full, x_full_sparse, x_uncond, t_tensor)

            estimated_noise = est_noise_zero_cond + self.w_uncond * (
                    est_noise_cond - est_noise_zero_cond)

            x_t_minus_one = self.dpm_scheduler.step(estimated_noise, t, x_full.F.reshape(1, -1, 3)).prev_sample
            x_full = self.points_to_tensor(x_t_minus_one)

            x_cond, x_uncond = self.reset_partial_pcd(x_cond)
            torch.cuda.empty_cache()

        return x_full.F.cpu().detach().numpy()


def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1, 4))[:, :3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(
            f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")


def safe_create_directory(dir_path, timestamp):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        if os.listdir(dir_path):
            print(f'Directory {dir_path} already exists, creating one with a timestamp.')
            os.makedirs(f'{dir_path}_{timestamp}')
            return f'{dir_path}_{timestamp}'
        else:
            return dir_path
    else:
        os.makedirs(dir_path)
        return dir_path


def prepare_outputs(config, timestamp, selected_files):
    full_exp_dir = join(config['output_folder'], config['exp_name'])
    os.makedirs(full_exp_dir, exist_ok=True)
    save_config(config, join(full_exp_dir, f'config_{timestamp}.yaml'))
    diff_folder = safe_create_directory(join(full_exp_dir, 'diff'), timestamp)

    with open(join(full_exp_dir, f'filenames.txt'), 'w') as f:
        for file_path in selected_files:
            f.write(file_path + "\n")
    return diff_folder, full_exp_dir


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--diff', '-d', type=str, help='path to the scan sequence')
@click.option('--denoising_steps', '-T', type=int, help='number of denoising steps (default: 50)')
@click.option('--cond_weight', '-s', type=float, help='conditioning weight (default: 6.0)')
@click.option('--exp_name', type=str, help='exp name')
@click.option('--start_index', type=int, help='start index')
@click.option('--sequence', type=str, help='sequence')
@click.option('--seed', type=int, help='seed')
@click.option('--end_index', type=int, help='end index')
@click.option('--starting_point', '-t0', type=int, help='starting point t0 of diffusion')
def main(config_path, **kwargs):
    ctx = click.get_current_context()
    config = smart_config(config_path, ctx)

    seed_everything(config.get('seed', 42), workers=True)
    # Handle if there is no exp_name in the config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config['exp_name'] is None:
        config['exp_name'] = f'exp_{timestamp}'
    sequence = f"{int(config['sequence']):02}"

    if config['minival']:
        with open(config['canonical_minival_filename'], 'r') as file:
            selected_files = [line.strip() for line in file]
    else:
        velodyne_dir = join(config['dataset_path'], 'dataset/sequences', sequence, 'velodyne')
        if config['end_index'] is None:
            config['end_index'] = len(os.listdir(velodyne_dir))
        selected_files = sorted(glob(join(velodyne_dir, '*.bin')))[config['start_index']: config['end_index']]

    diff_folder, full_exp_dir = prepare_outputs(config, timestamp, selected_files)

    total_time = 0

    for pcd_path in tqdm.tqdm(selected_files):
        str_index = pcd_path.split('/')[-1].split('.')[0]

        print(f"Now processing: {str_index + '.ply'}")
        diff_completion = DiffCompletion(config, full_exp_dir)

        points = load_pcd(pcd_path)

        start = time.time()
        diff_scan = diff_completion.denoise_scan(points)
        end = time.time()
        print(f'took: {end - start}s')
        total_time += end - start

        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(diff_scan)
        pcd_diff.estimate_normals()
        o3d.io.write_point_cloud(
            join(
                diff_folder, str_index + '.ply'
            ), pcd_diff
        )

    print(f'took in total: {total_time}s')


if __name__ == '__main__':
    main()
