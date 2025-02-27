import logging
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed
from analysis import ANALYSIS_TASKS


class LimitedAvgMeter(object):
    """
    A limited average meter that keeps track of the last 'max_num' values.

    Args:
        max_num (int, optional): Maximum number of values to keep. Defaults to 10.
    """
    def __init__(self, max_num=10):
        self.avg = 0.0
        self.num_list = []
        self.max_num = max_num

    def append(self, x):
        """
        Append a new value to the meter and update the average.

        Args:
            x (float): New value to append.
        """
        self.num_list.append(x)
        len_list = len(self.num_list)
        if len_list > 0:
            if len_list < self.max_num:
                self.avg = sum(self.num_list) / len_list
            else:
                self.avg = sum(self.num_list[len_list - self.max_num:len_list]) / self.max_num


class CallBackVerification(object):
    """
    A callback class for verification during training.

    Args:
        val_targets (list): List of validation targets.
        rec_prefix (str): Prefix for the validation dataset.
        summary_writer (SummaryWriter, optional): TensorBoard summary writer. Defaults to None.
        image_size (tuple, optional): Size of the input images. Defaults to (112, 112).
    """
    def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112)):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        """
        Perform verification test on the validation datasets.

        Args:
            backbone (torch.nn.Module): Backbone network.
            global_step (int): Current global step.
        """
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        """
        Initialize the validation datasets.

        Args:
            val_targets (list): List of validation targets.
            data_dir (str): Directory containing the validation datasets.
            image_size (tuple): Size of the input images.
        """
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        """
        Call the verification callback.

        Args:
            num_update (int): Current update number.
            backbone (torch.nn.Module): Backbone network.
        """
        if self.rank is 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    """
    A callback class for logging training progress.

    Args:
        frequent (int): Frequency of logging.
        total_step (int): Total number of steps.
        batch_size (int): Batch size.
        start_step (int, optional): Starting step. Defaults to 0.
        writer (SummaryWriter, optional): TensorBoard summary writer. Defaults to None.
    """
    def __init__(self, frequent, total_step, batch_size, start_step=0, writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 recognition_loss: AverageMeter,
                 analysis_losses: List,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        """
        Log training progress.

        Args:
            global_step (int): Current global step.
            loss (AverageMeter): Average loss meter.
            recognition_loss (AverageMeter): Average recognition loss meter.
            analysis_losses (List): List of average analysis loss meters.
            epoch (int): Current epoch.
            fp16 (bool): Whether using FP16 training.
            learning_rate (float): Current learning rate.
            grad_scaler (GradScaler): Gradient scaler for FP16 training.
        """
        if self.rank == 0 and global_step > 0 and (global_step + 1) % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec / 3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                    self.writer.add_scalar('recognition_loss', recognition_loss.avg, global_step)

                    for j in range(42):
                        self.writer.add_scalar(ANALYSIS_TASKS[j] + ' Training Loss', analysis_losses[j].avg,
                                               global_step)

                msg = "Speed %.2f samples/sec   Loss %.4f   Recognition Loss %.4f   " % (
                          speed_total, loss.avg, recognition_loss.avg)
                for j in range(42):
                    temp = ANALYSIS_TASKS[j] + " Loss %.4f   " % (analysis_losses[j].avg) + "   "
                    msg += temp

                if fp16:
                    temp = "LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                           "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              learning_rate, epoch, global_step, grad_scaler.get_scale(), time_for_end)
                else:
                    temp = "LearningRate %.6f   Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                               learning_rate, epoch, global_step, time_for_end)
                msg += temp
                msg += "\n\n"

                logging.info(msg)
                loss.reset()
                recognition_loss.reset()
                for each in analysis_losses:
                    each.reset()
                
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()