import typing as t
import torch
import numpy as np
import colorsys
from PIL import Image, ImageFont
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


def detection_collate(batch: t.Iterable[t.Tuple]) -> t.Tuple[torch.Tensor, t.List, t.List]:
    """
    custom collate func for dealing with batches of images that have a different number
    of object annotations (bbox).

    by the way, this func is used to customize the content returned by the dataloader.
    """

    labels = []
    images = []
    img_paths = []
    for img, label, img_path in batch:
        images.append(img)
        labels.append(label)
        img_paths.append(img_path)
    return torch.stack(images, dim=0), labels, img_paths


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


class ComputeMAP(object):
    """
    compute map
    """

    def __init__(self, iou_thresh: float = 0.5):
        self.iou_thresh = iou_thresh

    def calculate_tp(
            self,
            pred_coord: torch.Tensor,
            pred_score: torch.Tensor,
            gt_coord: torch.Tensor,
            gt_difficult: torch.Tensor,
    ) -> t.Tuple[int, t.List, t.List]:
        """

        calculate tp/fp for all predicted bboxes for one class of one image.

        Input:
            pred_coord: Tensor -> [N, 4], coordinates of all prediction boxes for a certain category
                        in a certain image(x0, y0, x1, y1)
            pred_score: Tensor -> [N, 1], score(confidence) of all prediction boxes for a certain category
                        in a certain image
            gt_coord: Tensor -> [M, 4], coordinates of all prediction boxes for a certain category
                        in a certain image(x0, y0, x1, y1)
            gt_difficult: Tensor -> [M, 1] -> whether the value of gt box of a certain category
                        in a certain image is difficult target?
            iou_thresh: the threshold to split TP and FP/FN.

        Output:
            gt_num: the number of gt box for a certain category in a certain image
            tp_list:
            conf_list:
        """

        not_difficult_gt_mask = torch.LongTensor(gt_difficult == 0)
        gt_num = not_difficult_gt_mask.sum()
        if gt_num == 0 or gt_coord.numel() == 0:
            return 0, [], []

        if pred_coord.numel() == 0:
            return len(gt_coord), [], []

        # compute iou of gt-box and pred-box
        gt_size, pred_size = gt_coord.size(0), pred_coord.size(0)

        inter_t_l = torch.max(gt_coord[..., :2].unsqueeze(1).expand(gt_size, pred_size, 2),
                              pred_coord[..., :2].unsqueeze(0).expand(gt_size, pred_size, 2))
        inter_b_r = torch.min(gt_coord[..., 2:].unsqueeze(1).expand(gt_size, pred_size, 2),
                              pred_coord[..., 2:].unsqueeze(0).expand(gt_size, pred_size, 2))
        inter = torch.clamp(inter_b_r - inter_t_l, min=0)
        inter = inter[..., 0] * inter[..., 1]

        area_gt = ((gt_coord[..., 2] - gt_coord[..., 0]) * (gt_coord[..., 3] - gt_coord[..., 2])).unsqueeze(
            1).expand_as(inter)
        area_pred = ((pred_coord[..., 2] - pred_coord[..., 0]) * (pred_coord[..., 3] - pred_coord[..., 2])).unsqueeze(
            1).expand_as(inter)

        iou_plural = inter / (area_pred + area_gt - inter + 1e-20)

        max_iou_val, max_iou_idx = torch.max(iou_plural, dim=0)

        # remove/ignore difficult gt box and the corresponding pred iou
        not_difficult_pb_mask = iou_plural[not_difficult_gt_mask].max(dim=0)[0] == max_iou_val
        max_iou_val, max_iou_idx = max_iou_val[not_difficult_pb_mask], max_iou_idx[not_difficult_pb_mask]
        if max_iou_idx.numel() == 0:
            return gt_num, [], []

        # for different bboxes that match to the same gt, set the highest score tp=1, and the other tp=0
        # score = conf * iou
        conf = pred_score.view(-1)[not_difficult_pb_mask]
        tp_list = torch.zeros_like(max_iou_val)
        for i in max_iou_idx[max_iou_val > self.iou_thresh].unique():
            gt_mask = (max_iou_val > self.iou_thresh) * (max_iou_idx == i)
            idx = (conf * gt_mask.float()).argmax()
            tp_list[idx] = 1
        return gt_num, tp_list.tolist(), conf.tolist()

    def calculate_pr(self, gt_num: int, tp_list: t.List, confidence_score: t.List) -> t.Tuple[t.List, t.List]:
        """
        calculate p-r according to gt number and tp_list for a certain category in a certain image
        """
        if gt_num == 0 or len(tp_list) == 0 or len(confidence_score) == 0:
            return [0], [0]
        if isinstance(tp_list, (tuple, list)):
            tp_list = np.array(tp_list)
        if isinstance(confidence_score, (tuple, list)):
            confidence_score = np.array(confidence_score)

        assert len(tp_list) == len(confidence_score), 'the length of tp_list is not equal to that in confidence score'

        # sort from max to min
        sort_mask = np.argsort(-confidence_score)
        tp_list = tp_list[sort_mask]

        # x = [1,3,1,2,5] -> np.cumsum(x) -> [1,4,5,7,12] ==> prefix sum
        recall = np.cumsum(tp_list) / gt_num
        precision = np.cumsum(tp_list) / (np.arange(len(tp_list)) + 1)

        return recall.tolist(), precision.tolist()


def set_font_thickness(font_filename: str, size: int, thickness: int):
    """
    set font and thickness of draw
    """
    font = ImageFont.truetype(font=font_filename, size=size)
    thickness = max(thickness, 1)
    return font, thickness


def generate_colors(classes_num: int) -> t.List:
    """
    generate different kinds of color according to class num
    """
    hsv_tuples = [(x / classes_num, 1., 1.) for x in range(classes_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


if __name__ == "__main__":
    file_path = r'C:\Users\24717\Projects\pascal voc2012\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg'
    image_ = Image.open(file_path)
    print(image_.size)
    image_.show()
