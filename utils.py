import typing as t

import torch
from PIL import Image


def resize_img(image: t.Any, new_size: t.Tuple[int, int], letterbox_image: bool):
    """
    resize img, add padding to picture <==> scale and paste
    """
    o_w, o_h = image.size
    n_w, n_h = new_size
    if letterbox_image:
        # no deformed conversion
        scale = min(n_w / o_w, n_h / o_h)
        n_w_ = int(o_w * scale)
        n_h_ = int(o_h * scale)

        image = image.resize((n_w_, n_h_), Image.BICUBIC)
        new_image = Image.new('RGB', new_size, (128, 128, 128))
        new_image.paste(image, ((n_w - n_w_) // 2, (n_h - n_h_) // 2))
    else:
        new_image = image.resize((n_w, n_h), Image.BICUBIC)
    return new_image


def compute_iou_gt_anchors(
        gt_boxes: torch.Tensor[torch.Tensor],
        anchor_boxes: torch.Tensor[torch.Tensor],
) -> torch.Tensor[torch.Tensor]:
    """
    Calculate the iou of anchors(dimension is [b,4]) and gt boxes(dimension is [9,4])
    Input:
            anchor_boxes: Tensor -> [b, 4]
            gt_boxes: Tensor -> [9, 4]
    Output:
            iou_plural: Tensor -> [b, 9]

    note: b means the number of box in one sample
    """

    a_x1, a_y1 = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2, anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2
    a_x2, a_y2 = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2, anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2

    gt_x1, gt_y1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2, gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gt_x2, gt_y2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2, gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    # store top-left and bottom-right coordinate
    box_a, box_gt = torch.zeros(anchor_boxes), torch.zeros(gt_boxes)
    box_a[:, 0], box_a[:, 1], box_a[:, 0], box_a[:, 1] = a_x1, a_y1, a_x2, a_y2
    box_gt[:, 0], box_gt[:, 1], box_gt[:, 0], box_gt[:, 1] = gt_x1, gt_y1, gt_x2, gt_y2

    a_size, gt_size = anchor_boxes.size(0), gt_boxes.size(0)

    # compute intersection
    # [b, 2] -> [b, 9, 2]
    inter_t_l = torch.max(box_gt[:, :2].unsqueeze(1).expand(gt_size, a_size, 2),
                          box_a[:, :2].unsqueeze(0).expand(gt_size, a_size, 2))
    inter_b_r = torch.min(box_gt[:, 2:].unsqueeze(1).expand(gt_size, a_size, 2),
                          box_a[:, 2:].unsqueeze(0).expand(gt_size, a_size, 2))
    # compress negative numbers to 0
    inter = torch.clamp(inter_b_r - inter_t_l, min=0)
    inter = inter[..., 0] * inter[..., 1]

    # compute union
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    a_area = anchor_boxes[:, 2] * anchor_boxes[:, 3]

    return inter / (gt_area + a_area - inter + 1e-20)


if __name__ == "__main__":
    file_path = r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg'
    image_ = Image.open(file_path)

    image_ = resize_img(image_, (416, 416), True)
    image_.show()
