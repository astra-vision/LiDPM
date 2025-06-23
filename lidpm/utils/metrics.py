import numpy as np
import scipy
from chamferdist import ChamferDistance
import torch
from torchmetrics import Metric


# region torchMetrics
class PrecisionRecall(Metric):
    def __init__(self, min_t: float, max_t: float, num: int):
        super().__init__()
        self.thresholds = torch.tensor(np.linspace(min_t, max_t, num))
        for idx in range(num):
            self.add_state(f"pr_{str(idx)}", default=torch.tensor([], dtype=torch.float32),
                           dist_reduce_fx="cat")
            self.add_state(f"re_{str(idx)}", default=torch.tensor([], dtype=torch.float32),
                           dist_reduce_fx="cat")
            self.add_state(f"f1_{str(idx)}", default=torch.tensor([], dtype=torch.float32),
                           dist_reduce_fx="cat")
        self.ch = ChamferDistance()
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, gt_pcd: torch.Tensor, pt_pcd: torch.Tensor):
        batch_size = pt_pcd.shape[0]
        self.count += batch_size

        dist_pt_2_gt = self.ch(pt_pcd, gt_pcd, point_reduction=None, batch_reduction=None)
        dist_gt_2_pt = self.ch(gt_pcd, pt_pcd, point_reduction=None, batch_reduction=None)

        for idx, t in enumerate(self.thresholds):
            p = (dist_pt_2_gt < t).float().mean(1) * 100
            r = (dist_gt_2_pt < t).float().mean(1) * 100

            f = torch.where((p + r) != 0, (2 * p * r) / (p + r), torch.tensor(0.0, device=p.device))

            setattr(self, f"pr_{str(idx)}", torch.cat([getattr(self, f"pr_{str(idx)}"), p.detach()]))
            setattr(self, f"re_{str(idx)}", torch.cat([getattr(self, f"re_{str(idx)}"), r.detach()]))
            setattr(self, f"f1_{str(idx)}", torch.cat([getattr(self, f"f1_{str(idx)}"), f.detach()]))

    def compute(self):
        dx = self.thresholds[1] - self.thresholds[0]
        perfect_predictor = scipy.integrate.simpson(torch.ones_like(self.thresholds), dx=dx)

        pr, re, f1 = self.compute_at_all_thresholds()

        pr_area = scipy.integrate.simpson(pr.detach().cpu(), dx=dx)
        norm_pr_area = pr_area / perfect_predictor

        re_area = scipy.integrate.simpson(re.detach().cpu(), dx=dx)
        norm_re_area = re_area / perfect_predictor

        f1_area = scipy.integrate.simpson(f1.detach().cpu(), dx=dx)
        norm_f1_area = f1_area / perfect_predictor

        return norm_pr_area, norm_re_area, norm_f1_area

    def compute_at_all_thresholds(self):
        pr = torch.stack([getattr(self, f"pr_{str(idx)}").mean() for idx in range(len(self.thresholds))])
        re = torch.stack([getattr(self, f"re_{str(idx)}").mean() for idx in range(len(self.thresholds))])
        f1 = torch.stack([getattr(self, f"f1_{str(idx)}").mean() for idx in range(len(self.thresholds))])
        return pr, re, f1


class ChamferDistanceMetric(Metric):
    def __init__(self, mode='norm_squared'):
        super(ChamferDistanceMetric, self).__init__()
        self.add_state("dists_gt_to_pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("dists_pred_to_gt", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("dists", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ch = ChamferDistance().to(self.device)
        self.mode = mode  # ['norm_squared' or 'norm']

    def update(self, gt_pcd, pred_pcd):
        batch_size = pred_pcd.shape[0]
        if self.mode == 'norm_squared':
            dist_pt_2_gt = self.ch(pred_pcd, gt_pcd, point_reduction="mean", batch_reduction="sum")
            dist_gt_2_pt = self.ch(gt_pcd, pred_pcd, point_reduction="mean", batch_reduction="sum")

            self.dists_gt_to_pred += dist_gt_2_pt
            self.dists_pred_to_gt += dist_pt_2_gt
            self.dists += (dist_gt_2_pt + dist_pt_2_gt)
        elif self.mode == 'norm':
            dist_pt_2_gt = self.ch(pred_pcd, gt_pcd, point_reduction=None, batch_reduction=None)
            dist_gt_2_pt = self.ch(gt_pcd, pred_pcd, point_reduction=None, batch_reduction=None)

            dist_pt_2_gt = torch.sqrt(dist_pt_2_gt)
            dist_gt_2_pt = torch.sqrt(dist_gt_2_pt)

            self.dists_gt_to_pred += dist_gt_2_pt.mean(-1).sum(0)
            self.dists_pred_to_gt += dist_pt_2_gt.mean(-1).sum(0)
            self.dists += (dist_gt_2_pt.mean(-1).sum(0) + dist_pt_2_gt.mean(-1).sum(0)) / 2

        self.count += batch_size

    def compute(self):
        gt2pred = self.dists_gt_to_pred / self.count
        pred2gt = self.dists_pred_to_gt / self.count
        ch_bidirect = self.dists / self.count
        return {"gt2pred": gt2pred, "pred2gt": pred2gt, "ch_bidirect": ch_bidirect}

# endregion
