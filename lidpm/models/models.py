import torch
import torch.nn.functional as F
import lidpm.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
from lidpm.utils.scheduling import beta_func
from tqdm import tqdm
import os
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from lidpm.utils.collations import feats_to_coord
from lidpm.utils.metrics import ChamferDistanceMetric, PrecisionRecall


class DiffusionPoints(LightningModule):
    def __init__(self, hparams: dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['beta_start'],
                self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.t_steps_train = self.hparams['diff']['t_steps_train']
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=self.device
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=self.device
        )

        self.betas = torch.tensor(self.betas, device=self.device)
        self.alphas = torch.tensor(self.alphas, device=self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)

        self.partial_enc = minknet.MinkGlobalEncIN(
            in_channels=self.hparams['model']['in_dim'], out_channels=self.hparams['model']['out_dim']
        )
        self.model = minknet.MinkUNetDiffIN(
            in_channels=self.hparams['model']['in_dim'], out_channels=self.hparams['model']['out_dim']
        )

        self.val_chamfer_distance = ChamferDistanceMetric(mode="norm")
        self.precision_recall = PrecisionRecall(
            self.hparams['data']['resolution'], 2 * hparams['data']['resolution'],
            self.hparams['train']['thresholds_num']
        )

        self.uncond_w = self.hparams['train']['uncond_w']
        self.range_limits = self.hparams['data']['range_limits'][self.hparams['data']['range']]

        self.use_ddp = self.hparams['use_ddp']
        self.duplication_factor = self.hparams['data']['duplication_factor']
        self.num_points = self.hparams['data']['num_points']


    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:, None, None].to(self.device) * x + \
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None].to(self.device) * noise

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.uncond_w * (x_cond - x_uncond)

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = self.model(x_full, x_full_sparse, part_feat, t)
        num_feat = out.shape[-1]
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0], -1, num_feat)

    def points_to_tensor(self, x_feats, bs):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device) # [B * 180k, 1 + dim]

        x_coord = x_feats.clone()
        x_coord[:, 1:] = feats_to_coord(x_feats[:, 1:], self.hparams['data']['resolution'], bs)

        x_t = ME.TensorField(
            features=x_feats[:, 1:],
            coordinates=x_coord, #spatial locations of the input points (voxels), should come in shape [N, 4], where 4 is batch_index + x + y + z
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def training_step(self, batch: dict, batch_idx):
        torch.cuda.empty_cache()

        pcd_shape = batch['pcd_full'].shape
        bs = pcd_shape[0]
        noise = torch.randn(pcd_shape, device=self.device)  # initial random noise
        t = torch.randint(0, self.t_steps_train, size=(bs,)).to(self.device)  # sample step t
        t_sample = self.q_sample(batch['pcd_full'], t, noise)  # sample p_m [step t]

        x_full = self.points_to_tensor(t_sample, bs)


        # # for classifier-free guidance switch between conditional and unconditional training
        if torch.rand(1) > self.hparams['train']['uncond_prob']:
            scaled_pcd_part = self.q_sample(batch['pcd_part'], t, torch.zeros_like(batch['pcd_part']))
            scaled_x_part = self.points_to_tensor(scaled_pcd_part, bs)
        else:
            scaled_x_part = self.points_to_tensor(torch.zeros_like(batch['pcd_part']), bs)

        denoise_t = self.forward(x_full, x_full.sparse(), scaled_x_part, t)

        loss = self.p_losses(denoise_t, noise)
        squared_error = (denoise_t - noise) ** 2
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/var', squared_error.var(), on_step=True, on_epoch=True)
        self.log('train/std', squared_error.std(), on_step=True, on_epoch=True)

        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch: dict, batch_idx):
        pcd_shape = batch['pcd_full'].shape
        bs = pcd_shape[0]
        os.makedirs(f'{self.logger.log_dir}/val_pcds/', exist_ok=True)

        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            noise = torch.randn(pcd_shape, device=self.device)  # initial random noise
            t = torch.ones(bs).long().to(self.device) * (self.t_steps_train - 1)
            q_sampled_noised_pcd = self.q_sample(batch['pcd_full'], t, noise)  # sample p_m [step t]

            # replace the original points with the noise sampled
            x_full = self.points_to_tensor(q_sampled_noised_pcd, bs)

            if self.hparams['train']['uncond_prob'] < 1.:
                x_cond = self.points_to_tensor(
                    batch['pcd_part'], bs
                )
            else:
                x_cond = None
            # x_cond = None

            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), bs
            )

            x_gen_eval = self.p_sample_loop(x_full, x_cond, x_uncond, bs)
            x_gen_eval = x_gen_eval.F.reshape((bs, -1, 3)).detach()

            # save x_gen_eval as separate .txt pointclouds
            for (pcd_gen, pcd_gt, noised_pdc, filename) in zip(x_gen_eval, gt_pts, q_sampled_noised_pcd, batch['filename']):
                idx = filename.split('/')[-1].split('.')[0]
                np.savetxt(f'{self.logger.log_dir}/val_pcds/{idx}_gt.txt',
                           pcd_gt)
                np.savetxt(f'{self.logger.log_dir}/val_pcds/{idx}_gen_epoch_{self.current_epoch}.txt',
                           pcd_gen.detach().cpu().numpy())
                np.savetxt(f'{self.logger.log_dir}/val_pcds/{idx}_noised_epoch_{self.current_epoch}.txt',
                           noised_pdc.detach().cpu().numpy())

        self.precision_recall.update(batch['pcd_full'], x_gen_eval)

        self.val_chamfer_distance.update(batch['pcd_full'], x_gen_eval)
        torch.cuda.empty_cache()

    def reset_partial_pcd(self, x_part, bs):
        x_part = self.points_to_tensor(x_part.F.reshape(bs, -1, 3).detach(), bs)
        x_uncond = self.points_to_tensor(torch.zeros_like(x_part.F.reshape(bs, -1, 3)), bs)
        return x_part, x_uncond

    def p_sample_loop(self, x_full, x_part, x_zeros, bs):

        for t in tqdm(range(self.t_steps_train - 1, -1, -1)):
            scale_factor_noise = self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t]
            scale_factor = self.sqrt_recip_alphas[t]
            sigma_t = torch.sqrt(self.betas)[t]
            noise = torch.randn((bs, self.num_points, 3), device=self.device)

            t = torch.ones(bs).long().to(self.device) * t
            x_full_sparse = x_full.sparse()

            if x_part:
                est_noise_cond = self.forward(x_full, x_full_sparse, x_part, t)
                est_noise_zero_cond = self.forward(x_full, x_full_sparse, x_zeros, t)
                estimated_noise = est_noise_zero_cond + self.hparams['train']['uncond_w'] * (est_noise_cond - est_noise_zero_cond)
            else:
                estimated_noise = self.forward(x_full, x_full_sparse, x_zeros, t)


            x_t_minus_one = scale_factor * (
                    x_full.F.reshape(bs, -1, 3) - scale_factor_noise * estimated_noise
            ) + sigma_t * noise
            x_full = self.points_to_tensor(x_t_minus_one.detach(), bs)

            if x_part:
                x_part, x_zeros = self.reset_partial_pcd(x_part, bs)
            torch.cuda.empty_cache()

        return x_full

    def validation_epoch_end(self, outputs):
        # Compute the Chamfer distance metrics
        chamfer_metrics = self.val_chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute()

        # Log metrics
        self.log("val/chamfer_gt2pred_norm", chamfer_metrics['gt2pred'], on_epoch=True)
        self.log("val/chamfer_pred2gt_norm", chamfer_metrics['pred2gt'], on_epoch=True)
        self.log("val/chamfer_symmetric_norm", chamfer_metrics['ch_bidirect'], on_epoch=True)

        self.log('val/precision', pr, on_epoch=True)
        self.log('val/recall', re, on_epoch=True)
        self.log('val/fscore', f1, on_epoch=True)

        # Reset the metric for the next epoch
        self.val_chamfer_distance.reset()
        self.precision_recall.reset()

    def configure_optimizers(self):
        effective_bs = self.hparams['train']['n_gpus'] * self.hparams['train']['batch_size']
        multiplier = effective_bs / 2
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams['train']['lr'] * (multiplier ** 0.5),
            betas=(
                1. - multiplier * (1. - self.hparams['train']['beta1']),
                1. - multiplier * (1. - self.hparams['train']['beta2']),
            ),
            eps=self.hparams['train']['eps'] / (multiplier ** 0.5)
        )

        return optimizer
