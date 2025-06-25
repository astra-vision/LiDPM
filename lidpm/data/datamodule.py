from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from lidpm.data.SemanticKITTI import KITTIDataSet
from lidpm.utils.collations import SparseSegmentCollation
import warnings

warnings.filterwarnings('ignore')

__all__ = ['SemKittiDataModule']

class SemKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = KITTIDataSet(
            data_dir=self.cfg['data']['data_dir'],
            gt_map_dir=self.cfg['data']['gt_map_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            range_limits=self.cfg['data']['range_limits'][self.cfg['data']['range']],
            duplication_factor=self.cfg['data']['duplication_factor'],
            minival_path=self.cfg['data']['canonical_minival'],
            num_validation_pointclouds=self.cfg['data']['num_validation_pointclouds']

        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'],
                            collate_fn=collate, drop_last=True, pin_memory=True)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = KITTIDataSet(
            data_dir=self.cfg['data']['data_dir'],
            gt_map_dir=self.cfg['data']['gt_map_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            range_limits=self.cfg['data']['range_limits'][self.cfg['data']['range']],
            duplication_factor=self.cfg['data']['duplication_factor'],
            minival_path=self.cfg['data']['canonical_minival'],
            num_validation_pointclouds=self.cfg['data']['num_validation_pointclouds']

        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'],
                            collate_fn=collate, pin_memory=True)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = KITTIDataSet(
            data_dir=self.cfg['data']['data_dir'],
            gt_map_dir=self.cfg['data']['gt_map_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            range_limits=self.cfg['data']['range_limits'][self.cfg['data']['range']],
            duplication_factor=self.cfg['data']['duplication_factor'],
            minival_path=self.cfg['data']['canonical_minival'],
            num_validation_pointclouds=self.cfg['data']['num_validation_pointclouds']

        )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'],
                            collate_fn=collate, pin_memory=True)
        return loader

dataloaders = {
    'KITTI': SemKittiDataModule,
}

