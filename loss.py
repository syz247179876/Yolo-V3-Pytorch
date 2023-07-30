import torch.nn as nn
import torch
import typing as t

from argument import args_train
from settings import *
from utils import compute_iou_gt_anchors


class YoloV3Loss(nn.Module):

    def __init__(
            self,
            anchors: [t.List[t.List]] = ANCHORS,
            classes_num: int = VOC_CLASS_NUM,
            input_shape: t.Tuple[int, int] = (416, 416),
    ):
        super(YoloV3Loss, self).__init__()
        self.anchors = anchors
        self.classes_num = classes_num
        self.input_shape = input_shape
        self.opts = args_train.opts
        self.use_gpu = args_train.opts.use_gpu
        self.gpu_id = args_train.opts.gpu_id

    def generator_labels(
            self,
            level: int,
            labels: torch.Tensor[torch.Tensor],
            scaled_anchors: t.List[t.Tuple],
            f_h: int,
            f_w: int
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        divide positive labels, negative labels, ignore labels

        Input:
            labels: t.List[torch.Tensor] -> [[[x_min, y_min, x_max, y_max], ...], [], []]
            x_min, y_min, x_max, y_max has been normalized
        """
        batch_size = len(labels)

        gt_tensor = torch.zeros(batch_size, ANCHORS_NUM, f_h, f_w, 5 + self.classes_num + 1, requires_grad=False)
        # make net pay attention to small obj
        box_loss_scale = torch.zeros(batch_size, ANCHORS_NUM, f_h, f_w, requires_grad=False)

        for batch_idx, batch_label in enumerate(labels):
            if len(batch_label) == 0:
                continue
            batch_tensor = torch.zeros_like(batch_label)
            # according to all labels in one batch, trans normalization to feature map size
            # batch_tensor[0:, [0,1,2,3,4]] => n_x_mid, n_y_mid, n_w, n_h , cls_id
            batch_tensor[:, [0, 2]] = batch_label[:, [0, 2]] * f_w
            batch_tensor[:, [1, 3]] = batch_label[:, [1, 3]] * f_h
            batch_tensor[:, 4] = batch_label[:, 4]
            batch_tensor = batch_tensor.cpu()

            """
            compute iou between anchors and labels, we only pay attention to the shape of w and h, not care center 
            position, as we can learn the offset continuously.
            """
            gt_box = torch.cat((torch.zeros((batch_tensor.size(0), 2)), batch_tensor[:, [2, 3]]), 1).type(torch.float32)
            anchors_box = torch.cat((torch.zeros(len(self.anchors), 2), self.anchors), 1).type(torch.float32)

            # TODO: can we use the anchors relative to the current feature layer, not all anchors?
            iou_plural = compute_iou_gt_anchors(gt_box, anchors_box)
            # positive samples , dim = [b, 1]
            best_iou_plural = torch.argmax(iou_plural, dim=-1)

            # generate positive, ignore and negative samples' label
            for i, a_idxes in enumerate(iou_plural):
                # best anchor not in the anchors relative to the current level feature map
                best_a_idx = best_iou_plural[i][0]
                for a_id in a_idxes:
                    aim_a_idx = ANCHORS_MASK[level].index(best_a_idx)
                    grid_x = torch.floor(batch_tensor[i, 0]).long()
                    grid_y = torch.floor(batch_tensor[i, 1]).long()
                    if a_id == best_a_idx and best_a_idx in ANCHORS_MASK[level]:
                        # access best anchor by using ANCHORS_MASK[level][aim_a_idx], 0 <= aim_a_dix <= 2

                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 0] = batch_tensor[i, 0] - grid_x.float()
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 1] = batch_tensor[i, 1] - grid_y.float()
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 2] = torch.log(
                            batch_tensor[i, 2] / scaled_anchors[best_a_idx][0])
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 3] = torch.log(
                            batch_tensor[i, 3] / scaled_anchors[best_a_idx][1])
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 4] = 1
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, 5 + batch_tensor[i, 4].long()] = 1
                        box_loss_scale[batch_idx, aim_a_idx, grid_y, grid_x] = batch_label[i, 2] * batch_label[i, 3]
                        # set positive samples
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, -1] = 1
                    elif iou_plural[i][a_id] < self.opts.anchors_thresh:
                        # set negative samples
                        gt_tensor[batch_idx, aim_a_idx, grid_y, grid_x, -1] = -1

        return gt_tensor, box_loss_scale

    def forward(self, level: int, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Input:
            level: it means the level of feature map, there are three level in YOLO v3 net, 13x13, 26x26, 52x52
        """
        conf_loss_func = nn.BCELoss(reduction='none')
        xy_loss_func = nn.BCELoss(reduction='none')
        wh_loss_func = nn.BCELoss(reduction='none')
        cls_loss_func = nn.CrossEntropyLoss(reduction='none')

        batch_size, _, f_w, f_h = pred.size()
        stride_h = self.input_shape[0] / f_h
        stride_w = self.input_shape[1] / f_w

        scaled_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in ANCHORS[level]]
        pred = pred.view(batch_size, ANCHORS_NUM, 4 + 1 + self.classes_num, f_h, f_w). \
            permute(0, 1, 3, 4, 2).contiguous()

        tx = torch.sigmoid(pred[..., 0])
        ty = torch.sigmoid(pred[..., 1])
        tw = pred[..., 2]
        th = pred[..., 3]

        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = pred[..., 5:]

        gt_tensor, box_loss_scale = self.generator_labels(level, labels, scaled_anchors, f_h, f_w)

        if self.use_gpu:
            gt_tensor = gt_tensor.to(self.gpu_id)
            # small obj has larger scale weight, larger obj has smaller scale weight
            box_loss_scale = (2 - box_loss_scale).to(self.gpu_id)

        avg_loss = torch.tensor(0.)

        # coordinate offset loss
        loss_tx = torch.mean(
            xy_loss_func(tx, gt_tensor[..., 0]) * gt_tensor[..., -1] * box_loss_scale * self.opts.coord_weight)
        loss_ty = torch.mean(
            xy_loss_func(ty, gt_tensor[..., 1]) * gt_tensor[..., -1] * box_loss_scale * self.opts.coord_weight)

        loss_tw = torch.mean(
            wh_loss_func(tw, gt_tensor[..., 2]) * gt_tensor[..., -1] * box_loss_scale * self.opts.coord_weight)
        loss_th = torch.mean(
            wh_loss_func(th, gt_tensor[..., 3]) * gt_tensor[..., -1] * box_loss_scale * self.opts.coord_weight)

        loss_cls = torch.mean(cls_loss_func(pred_cls, gt_tensor[..., 5:]) * gt_tensor[..., -1])

        loss_conf = torch.mean(conf_loss_func(pred_conf[gt_tensor[..., 4] == 1], gt_tensor[gt_tensor[..., 4] == 1]))
        no_obj_loss_conf = torch.mean(
            conf_loss_func(pred_conf[gt_tensor[..., 4] == -1], gt_tensor[gt_tensor[..., 4] == -1]))
        avg_loss += loss_tx + loss_ty + loss_tw + loss_th + loss_cls + loss_conf + no_obj_loss_conf

        return avg_loss
