import os.path
import time
import typing as t
import numpy as np
import torch
from PIL import Image, ImageDraw
from colorama import Fore
from torch.utils.data import DataLoader

from argument import args_test
from data.voc_data import VOCDataset
from decode import DecodeFeature
from loss import YoloV3Loss
from settings import VOC_CLASSES, INPUT_SHAPE
from utils import detection_collate, print_log, set_font_thickness, generate_colors


class YoloV3Test(object):
    """
    Yolo Model test
    """

    def __init__(
            self,
            img_size: int,
            classes_num: int,
            letterbox_image: bool = True
    ):
        self.opts = args_test.opts
        self.decode = DecodeFeature(img_size, classes_num)
        self.classes_num = classes_num
        self.letterbox_image = letterbox_image
        self.colors = generate_colors(classes_num)

    def main(self):
        test_dataset = VOCDataset(mode='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=detection_collate,
        )
        test_num = len(test_dataset)
        assert self.opts.pretrain_file is not None, 'need to load model trained!'
        model = torch.load(self.opts.pretrain_file)
        print_log(f'Load model file {self.opts.pretrain_file} successfully!')
        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)
        model.eval()

        loss_obj = YoloV3Loss()
        total_loss = 0.
        for batch, (x, labels, img_paths, image_shapes) in enumerate(test_loader):
            with torch.no_grad():
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                pred: t.Tuple = model(x)
                cur_loss = torch.tensor(0).float().to(self.opts.gpu_id)
                t1 = time.time()
                for idx, output in enumerate(pred):
                    cur_loss += loss_obj(idx, output, labels)
                t2 = time.time()
                total_loss += cur_loss.item()
                print_log(
                    f"cur batch: Average loss: {round(cur_loss.item(), 6)}, Inference time:{int((t2 - t1) * 1000)}ms, "
                    f"cur num: {len(x)}")

                # decode pred and execute multi-nms
                decoded_pred: t.List = self.decode.decode_pred(pred)
                results: t.List = self.decode.execute_nms(
                    torch.cat(decoded_pred, dim=1),
                    image_shapes,
                    self.letterbox_image
                )

                # set font and frame
                font_path = os.path.join(os.path.dirname(__file__), r'static/font/simhei.ttf')
                thickness = (image_shapes // np.mean(INPUT_SHAPE)).flatten()
                font, thickness = set_font_thickness(font_path,
                                                     np.floor(3e-2 * image_shapes[batch][1] + 0.5).astype('int32'),
                                                     thickness)

                top_cls_idx = np.array([result[:, 6] for result in results], dtype='int32')
                top_score = np.array([result[:, 4] * result[:, 5] for result in results], dtype='float')
                top_boxes = np.array([result[:, :4] for result in results], dtype='float')

                # rss = np.array([result[result[:, 4] * result[:, 5] > 1.] for result in results], dtype='float')

                # draw picture
                for cls_idx, score, boxes, img_path in zip(top_cls_idx, top_score, top_boxes, img_paths):
                    # dimension -> [num1, 1], num1 < anchor_num * g_h * g_w
                    cls_idx: np.ndarray
                    score: np.ndarray
                    boxes: np.ndarray
                    img_path: str
                    cur_img = Image.open(img_path)
                    width, height = cur_img.size

                    for i, s, b in zip(cls_idx, score, boxes):
                        pred_cls_name = VOC_CLASSES[int(i)]
                        top, left, bottom, right = b
                        top = max(0, np.floor(top).astype('int32'))
                        left = max(0, np.floor(left).astype('int32'))
                        bottom = min(cur_img.size[1], np.floor(bottom).astype('int32'))
                        right = min(cur_img.size[0], np.floor(right).astype('int32'))
                        draw = ImageDraw.Draw(cur_img)
                        label = f'{pred_cls_name} {round(s, 2)}'
                        label_size = draw.textsize(label, font)
                        print_log(f'{label} {top} {left} {bottom} {right}', Fore.BLUE)

                        if top - label_size[1] >= 0:
                            text_origin = np.array((left, top - label_size[1]))
                        else:
                            text_origin = np.array((left, top + 1))
                        if height > top > 0 and width > left > 0 and height > bottom > 0 and width > right > 0:
                            for step in range(thickness):
                                draw.rectangle((left + step, top + step, right - step, bottom - step),
                                               outline=self.colors[i])
                            draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)), fill=self.colors[i])
                            draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
                            del draw
                        # cv2.imshow("bbox", np.array(cur_img))
                        # cv2.waitKey(0)
                    cur_img.show()

        avg_loss = total_loss / (len(test_loader) + 1)
        print_log(f"Test set: Average loss: {round(avg_loss, 6)}, Total num: {test_num}")


if __name__ == '__main__':
    test = YoloV3Test(416, 20, )
    test.main()
