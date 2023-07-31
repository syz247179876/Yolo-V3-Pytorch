import typing as t
import torch
import numpy as np
from PIL import Image
from colorama import Fore


def resize_img_box(
        image: t.Any,
        labels: np.ndarray,
        new_size: t.Tuple[int, int],
        letterbox_image: bool
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    1.resize img, add padding to picture <==> scale and paste.
    2.resize each box in each sample according to new_size
    """
    o_w, o_h = image.size
    n_w, n_h = new_size

    if letterbox_image:
        # no deformed conversion
        scale = min(n_w / o_w, n_h / o_h)
        n_w_ = int(o_w * scale)
        n_h_ = int(o_h * scale)
        dw = (n_w - n_w_) // 2
        dh = (n_h - n_h_) // 2
        image = image.resize((n_w_, n_h_), Image.BICUBIC)
        new_image = Image.new('RGB', new_size, (128, 128, 128))
        new_image.paste(image, (dw, dh))

        if len(labels) > 0:
            # np.random.shuffle(labels)
            labels[:, [0, 2]] = labels[:, [0, 2]] * scale + dw
            labels[:, [1, 3]] = labels[:, [1, 3]] * scale + dh
            labels[:, 0: 2][labels[:, 0: 2] < 0] = 0
            labels[:, 2][labels[:, 2] > n_w] = n_w
            labels[:, 3][labels[:, 3] > n_h] = n_h
            box_w = labels[:, 2] - labels[:, 0]
            box_h = labels[:, 3] - labels[:, 1]
            # filter invalid box, which width and height of box less than 1.
            labels = labels[np.logical_and(box_w > 1, box_h > 1)]
    else:
        new_image = image.resize((n_w, n_h), Image.BICUBIC)
        # TODO resize box

    return np.array(new_image, np.float32), labels


def compute_iou_gt_anchors(
        gt_boxes: torch.Tensor,
        anchor_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the iou of gt boxes(dimension is [b,4]) and anchors(dimension is [9,4])
    Input:
            gt_boxes: Tensor -> [b, 4]
            anchor_boxes: Tensor -> [9, 4]
    Output:
            iou_plural: Tensor -> [b, 9]

    note: b means the number of box in one sample
    """

    a_x1, a_y1 = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2, anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2
    a_x2, a_y2 = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2, anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2

    gt_x1, gt_y1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2, gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gt_x2, gt_y2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2, gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    # store top-left and bottom-right coordinate
    box_a, box_gt = torch.zeros_like(anchor_boxes), torch.zeros_like(gt_boxes)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = a_x1, a_y1, a_x2, a_y2
    box_gt[:, 0], box_gt[:, 1], box_gt[:, 2], box_gt[:, 3] = gt_x1, gt_y1, gt_x2, gt_y2

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
    gt_area = (gt_boxes[:, 2] * gt_boxes[:, 3]).unsqueeze(1).expand_as(inter)
    a_area = (anchor_boxes[:, 2] * anchor_boxes[:, 3]).unsqueeze(0).expand_as(inter)

    return inter / (gt_area + a_area - inter + 1e-20)


def print_log(txt: str, color: t.Any = Fore.GREEN):
    print(color, txt)


def detection_collate(batch: t.Iterable[t.Tuple]):
    """
    custom collate func for dealing with batches of images that have a different number
    of object annotations (bbox).

    by the way, this func is used to customize the content returned by the dataloader.
    """

    labels = []
    images = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    return torch.stack(images, dim=0), labels


class ImageAugmentation(object):

    def __call__(self, image_path: str, labels: np.ndarray, input_shape=(416, 416)) -> t.Tuple[np.ndarray, np.ndarray]:
        image = Image.open(image_path)
        image.convert('RGB')
        # resize image and add grey on image, modify shape of ground-truth box (label) according to after-resized image
        image_data, labels = resize_img_box(image, labels, input_shape, True)
        return image_data, labels


class Normalization(object):

    def __init__(self, img_shape: t.Tuple[int, int] = (416, 416), mode='simple'):
        self.mode = mode
        self.img_shape = img_shape

    def __call__(self, images: np.ndarray, boxes: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        func = getattr(self, self.mode)
        return func(images, boxes)

    def simple(self, images: np.ndarray, boxes: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        simple normalization function, it directly divided the limit upper.
        for example
        1. for images, divide each pixel by 255
        2. for boxes, the center of the box divided by the width and height of the entire image
        """
        images = images / 255.0
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.img_shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.img_shape[0]
        return images, boxes


if __name__ == "__main__":
    file_path = r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg'
    image_ = Image.open(file_path)
    image_.show()
