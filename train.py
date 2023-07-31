import os.path
import time
import typing as t
import torch.nn as nn
import torch
from colorama import Fore
from torch.utils.data import DataLoader
from argument import args_train
from data.voc_data import VOCDataset
from loss import YoloV3Loss
from settings import ANCHORS
from utils import print_log, detection_collate
from model.darknet_53 import Darknet53


class YoloV3Train(object):

    def __init__(self):
        self.opts = args_train.opts

    def __train_epoch(
            self,
            model: Darknet53,
            loss_obj: YoloV3Loss,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            train_num: int,
    ) -> None:

        with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
            for batch, (x, labels) in enumerate(train_loader):

                if self.opts.use_gpu:
                    x = x.to(self.opts.gpu_id)
                    labels = labels.to(self.opts.gpu_id)

                pred = model(x)
                loss = loss_obj(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss = loss.item()

                if batch % self.opts.print_frequency == 0:
                    print_log("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f" %
                              (epoch, self.opts.end_epoch, batch, train_num // self.opts.batch_size, loss.item(),
                               avg_loss))
                    log_f.write("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f\n" %
                                (epoch, self.opts.end_epoch, batch, train_num // self.opts.batch_size, loss.item(),
                                 avg_loss))
                    log_f.flush()

    def __save_model(
            self,
            model: Darknet53,
            epoch: int,
    ) -> None:
        """
        save model
        """
        model_name = f'epoch{epoch}.pkl'
        torch.save(model, os.path.join(self.opts.checkpoints_dir, model_name))

    @property
    def init_lr(self) -> float:
        """
        adjust learning rate dynamically according to batch_size and epoch

        learning rate decrease five times every 30 epoch
        """
        max_lr = self.opts.lr_max
        min_lr = self.opts.lr_max
        batch_size = self.opts.batch_size
        lr = min(max(batch_size / 64 * self.opts.lr_base, min_lr), max_lr)
        return lr

    def main(self) -> None:
        """
        entrance of train
        """

        if not os.path.exists(self.opts.checkpoints_dir):
            os.mkdir(self.opts.checkpoints_dir)
        train_dataset = VOCDataset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=self.opts.shuffle,
            num_workers=self.opts.num_workers,
            drop_last=self.opts.drop_last,
            collate_fn=detection_collate,
        )
        train_num = len(train_dataset)

        if not self.opts.pretrain_file:
            model = YoloV3Loss()
            print_log('Init model successfully!')
        else:
            model = torch.load(self.opts.pretrain_file)
            print_log(f'Load model file {self.opts.pretrain_file} successfully!')
        loss_obj = YoloV3Loss()

        if self.opts.use_gpu:
            model.to(self.opts.gpu_id)

        # during train, use model.train() to update the mean and val according to each mini-batch of BN level
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.init_lr, momentum=0.9,
                                    weight_decay=self.opts.weight_decay)

        # adjust learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opts.decrease_interval, gamma=0.5)

        for e in range(self.opts.start_epoch, self.opts.end_epoch + 1):
            t1 = time.time()
            self.__train_epoch(model, loss_obj, train_loader, optimizer, e, train_num)
            t2 = time.time()
            scheduler.step()
            print_log("Training consumes %.2f second\n" % (t2 - t1), Fore.RED)
            with open(os.path.join(self.opts.checkpoints_dir, 'log.txt'), 'a+') as log_f:
                log_f.write(f'Training one epoch consumes %.2f second\n' % (t2 - t1))

                if e % self.opts.save_frequency == 0 or e == self.opts.end_epoch:
                    self.__save_model(model, e)


if __name__ == "__main__":
    yolo_v2 = YoloV3Train()
    yolo_v2.main()
