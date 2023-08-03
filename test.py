import time

import torch
from torch.utils.data import DataLoader

from argument import args_test
from data.voc_data import VOCDataset
from decode import DecodeFeature
from loss import YoloV3Loss
from utils import detection_collate, print_log


class YoloV3Test(object):
    """
    Yolo Model test
    """

    def __init__(self, img_size: int, classes_num: int):
        self.opts = args_test.opts
        self.decode = DecodeFeature(img_size, classes_num)

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
        with torch.no_grad():
            total_loss = 0.
            for batch, (x, labels) in enumerate(test_loader):
                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                pred = model(x)
                cur_loss = torch.tensor(0).float().to(self.opts.gpu_id)
                t1 = time.time()
                for idx, output in enumerate(pred):
                    cur_loss += loss_obj(idx, output, labels)
                t2 = time.time()
                total_loss += cur_loss.item()
                print_log(
                    f"cur batch: Average loss: {round(cur_loss.item(), 6)}, Inference time:{int((t2 - t1) * 1000)}ms, "
                    f"cur num: {len(x)}\n")

                # decode pred and execute multi-nms

                decoded_pred = self.decode.decode_pred(pred)
                self.decode.execute_nms()

            avg_loss = total_loss / (batch + 1)

            print_log(f"Test set: Average loss: {round(avg_loss, 6)}, Total num: {test_num}")


if __name__ == '__main__':
    test = YoloV3Test(416, 20, )
    test.main()
