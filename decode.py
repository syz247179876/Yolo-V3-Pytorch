"""
Decoding the extracted features
"""
import torch
import torch.nn as nn
import typing as t

from settings import *


class DecodeFeature(object):

    def __init__(self, img_size: int, classes_num: int, anchors_num: int):
        super(DecodeFeature, self).__init__()
        self.classes_num = classes_num
        self.anchors_num = anchors_num
        self.img_size = img_size

    def generator_xy_wh(
            self,
            grid_h: int,
            grid_w: int,
            batch_size: int,
            method: int = 1
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        generator grid and generate width and height according to the grid
        note: I implement two methods to generate grid and w, h

        1. the first method based on broadcast mechanism in Torch.
        2. the second method has the same dimensions as the feature map.
        """
        # method one:
        if method:
            grid_y, grid_x = torch.meshgrid((torch.arange(grid_h)), torch.arange(grid_w))
            # dimension -> [1, 1, 13, 13 ,2], if down sample multiple is 32
            grid_xy = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, grid_h, grid_w, 2)
        else:
            # method two:
            grid_x = torch.linspace(0, grid_h - 1, grid_h).repeat(grid_h, 1). \
                repeat(batch_size * ANCHORS_NUM, 1).view(batch_size, ANCHORS_NUM, grid_h, grid_w, 1)
            grid_y = torch.linspace(0, grid_w - 1, grid_w).repeat(grid_w, 1). \
                repeat(batch_size * ANCHORS_NUM, 1).t().view(batch_size, ANCHORS_NUM, grid_h, grid_w, 1)
            # dimension -> [B, anchors_nums, 13, 13, 2], if down sample multiple is 32
            grid_xy = torch.cat((grid_x, grid_y), dim=-1)
        return grid_xy, grid_xy

    def decode_pred(self, inputs: t.Union[t.List[torch.Tensor], t.Tuple[torch.Tensor, ...]]) -> t.List:
        """
        compute bx, by, bw, bh of bbox based on tx, ty, tw, th, and then replace tx, ty, tw, th in feature map
        Input:
            Tensor -> [B, self.anchors_num * (1 + 4 + self.classes_num), height, width]
        Output:
            Tensor -> [B, self.anchors_num, height, width, (1 + 4 + self.class_num)

            note: 4 -> [bx, by, bw, bn] are bbox's actual coordinate
        """
        output = []
        for idx, pred in enumerate(inputs):
            pred: torch.Tensor
            batch_size, _, g_h, g_w = pred.size()

            stride_h = self.img_size / g_h
            stride_w = self.img_size / g_w

            scaled_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in ANCHORS[idx]]

            # trans structure ot [B, self.anchor, height, width, (1 + 4 + self.class_num)]
            pred = pred.view(batch_size, ANCHORS_NUM, 4 + 1 + self.classes_num, g_h, g_w). \
                permute(0, 1, 3, 4, 2).contiguous()

            float_ = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor
            long_ = torch.cuda.LongTensor if pred.is_cuda else torch.LongTensor

            # compute bx, by, bw, bh
            grid_xy, anchor_wh = self.generator_xy_wh(g_h, g_w, batch_size)
            b_xy = torch.sigmoid(pred[..., :2]) + grid_xy

            pw, ph = pred[..., 2], pred[..., 3]
            anchor_w = float_(scaled_anchors).index_select(1, long_([0])).repeat(4, 1). \
                repeat(1, 1, g_h * g_w).view(pw.size())
            anchor_h = float_(scaled_anchors).index_select(1, long_([1])).repeat(4, 1). \
                repeat(1, 1, g_h * g_w).view(ph.size())
            b_w = torch.exp(pred[..., 2]) * anchor_w
            b_h = torch.exp(pred[..., 3]) * anchor_h

            pred[..., :2] = b_xy
            pred[..., 2] = b_w
            pred[..., 3] = b_h
            output.append(pred)
        return output


if __name__ == "__main__":
    d = DecodeFeature(416, 20, 3)
    in_p = [torch.randn(4, 75, 13, 13), torch.randn(4, 75, 26, 26), torch.randn(4, 75, 52, 52)]
    res = d.decode_pred(in_p)
    print(res[0].size(), res[1].size(), res[2].size())
