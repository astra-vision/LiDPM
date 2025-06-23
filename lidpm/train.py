import click
from os.path import join, dirname, abspath
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from utils.callbacks import TQDMProgressBar

import numpy as np
import torch
import yaml
import MinkowskiEngine as ME

import lidpm.data.datasets as datasets
import lidpm.models.models as models
import os


os.environ['OMP_NUM_THREADS'] = '11'

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--test', '-t', is_flag=True, help='test mode')
def main(config, weights, checkpoint, test):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    cfg = yaml.safe_load(open(config))
    seed_everything(cfg['train'].get('seed', 42), workers=True)

    cfg['use_ddp'] = (
                             cfg['train']['accelerator'] == "ddp"
                     ) and (torch.cuda.device_count() > 1)

    #Load data and model
    if weights is None: # training from scratch
        model = models.DiffusionPoints(cfg)
    else: # loading model from checkpoint
        if test:
            # we load the current config file just to overwrite inference parameters to try different stuff during inference
            ckpt_cfg = yaml.safe_load(open(weights.split('checkpoints')[0] + '/hparams.yaml'))
            ckpt_cfg['train']['uncond_min_w'] = cfg['train']['uncond_min_w']
            ckpt_cfg['train']['uncond_max_w'] = cfg['train']['uncond_max_w']
            ckpt_cfg['train']['num_workers'] = cfg['train']['num_workers']
            ckpt_cfg['train']['n_gpus'] = cfg['train']['n_gpus']
            ckpt_cfg['train']['batch_size'] = cfg['train']['batch_size']
            ckpt_cfg['data']['num_points'] = cfg['data']['num_points']
            ckpt_cfg['data']['data_dir'] = cfg['data']['data_dir']
            ckpt_cfg['experiment']['id'] = cfg['experiment']['id']
            ckpt_cfg['use_ddp'] = cfg['use_ddp']

            if 'dataset_norm' not in ckpt_cfg['data'].keys():
                ckpt_cfg['data']['dataset_norm'] = False
                ckpt_cfg['data']['std_axis_norm'] = False
            # if 'max_range' not in ckpt_cfg['data'].keys():
            #     ckpt_cfg['data']['max_range'] = 10.

            cfg = ckpt_cfg

        model = models.DiffusionPoints.load_from_checkpoint(weights, hparams=cfg)
        print(model.hparams)

    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)

    #region callbacks
    lr_monitor_step = LearningRateMonitor(logging_interval='step')

    checkpoint_epoch = ModelCheckpoint(
        auto_insert_metric_name=False,
        dirpath=f"checkpoints/{cfg['experiment']['id']}/{timestamp}",
        filename="epoch={epoch}-step={step}",
        every_n_epochs=cfg['train']['freq_ckpt'],
        save_last=True,
        save_top_k=-1,
        save_on_train_epoch_end=True
    )

    checkpoint_top_k = ModelCheckpoint(
        auto_insert_metric_name=False,
        dirpath=f"checkpoints/{cfg['experiment']['id']}/{timestamp}",
        filename="best-epoch={epoch}-step={step}-val_ch_sym={val/chamfer_symmetric_norm:.2f}",
        save_top_k=3,
        monitor="val/chamfer_symmetric_norm",
        mode="min"  # Assuming lower is better for 'val/cd_mean_epoch'
    )
    # endregion

    tb_logger = pl_loggers.TensorBoardLogger(
        join('experiments', cfg['experiment']['id'], timestamp),
        default_hp_metric=False
    )

    #Setup trainer
    if cfg['use_ddp']:
        print("Using DDP")
        cfg['train']['n_gpus'] = torch.cuda.device_count()
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        trainer = Trainer(gpus=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=[
                                lr_monitor_step, checkpoint_top_k, checkpoint_epoch,
                                TQDMProgressBar(),
                          ],
                          check_val_every_n_epoch=cfg['train']['check_val_every_n_epoch'],
                          num_sanity_val_steps=cfg['train']['num_sanity_val_steps'],
                          limit_val_batches=cfg['train']['limit_val_batches'],
                          limit_train_batches=cfg['train']['limit_train_batches'],
                          strategy='ddp',
                          accelerator='cuda',
                          )
    else:
        trainer = Trainer(gpus=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=[
                              lr_monitor_step, checkpoint_top_k, checkpoint_epoch,
                              TQDMProgressBar(),
                          ],
                          check_val_every_n_epoch=cfg['train']['check_val_every_n_epoch'],
                          num_sanity_val_steps=cfg['train']['num_sanity_val_steps'],
                          limit_val_batches=cfg['train']['limit_val_batches'],
                          limit_train_batches=cfg['train']['limit_train_batches']
                          )

    # Train!
    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data)

if __name__ == "__main__":
    main()
