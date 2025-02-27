# Import necessary libraries and modules
from timm.utils import accuracy, AverageMeter

import logging
import time
import torch
import torch.distributed as dist
import math

from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed
from .task_name import ANALYSIS_TASKS

# Function to reduce a tensor across all processes in a distributed setting
def reduce_tensor(tensor):
    """
    Reduces a tensor by summing it across all processes and then dividing by the number of processes.
    This is useful for synchronizing metrics across multiple GPUs.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

# Class to maintain a limited history of values and compute average and best values
class LimitedAvgMeter(object):
    """
    A class to maintain a limited history of values and compute average and best values.
    Useful for tracking metrics over a fixed window of recent updates.
    """
    def __init__(self, max_num=10, best_mode="max"):
        """
        Initializes the LimitedAvgMeter with a maximum number of values to store and a mode to determine the best value.
        """
        self.avg = 0.0
        self.num_list = []
        self.max_num = max_num
        self.best_mode = best_mode
        self.best = 0.0 if best_mode == "max" else 100.0

    def append(self, x):
        """
        Appends a new value to the history and updates the average and best values.
        """
        self.num_list.append(x)
        len_list = len(self.num_list)
        if len_list > 0:
            if len_list < self.max_num:
                self.avg = sum(self.num_list) / len_list
            else:
                self.avg = sum(self.num_list[(len_list - self.max_num):len_list]) / self.max_num

        if self.best_mode == "max":
            if self.avg > self.best:
                self.best = self.avg
        elif self.best_mode == "min":
            if self.avg < self.best:
                self.best = self.avg

# Class for verifying model performance on the CelebA dataset
class CelebAVerification(object):
    """
    A class for verifying model performance on the CelebA dataset.
    Tracks accuracy and loss for each of the 40 attributes.
    """
    def __init__(self, data_loader, summary_writer=None):
        """
        Initializes the CelebAVerification class with a data loader and an optional summary writer.
        """
        self.rank: int = distributed.get_rank()
        self.highest_acc_list = [0.0 for j in range(40)]
        self.highest_mean_acc = 0.0
        self.acc_list_corresponding_to_highest_mean_acc = []

        self.data_loader = data_loader
        self.summary_writer = summary_writer

        self.limited_meter = LimitedAvgMeter(best_mode="max")

    def ver_test(self, model, global_step):
        """
        Performs verification testing on the model for the CelebA dataset.
        Computes and logs accuracy and loss for each attribute.
        """
        logging.info("Val on CelebA:")

        criteria = [torch.nn.CrossEntropyLoss() for j in range(40)]
        loss_meters = [AverageMeter() for j in range(40)]
        acc_meters = [AverageMeter() for j in range(40)]

        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, targets) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)

            # Compute model output
            analysis_results = model(img)

            for j in range(40):
                analysis_target = targets[j].cuda(non_blocking=True)
                analysis_loss = criteria[j](analysis_results[j], analysis_target)
                analysis_acc, _ = accuracy(analysis_results[j], analysis_target, topk=(1, 1))

                analysis_loss = reduce_tensor(analysis_loss)
                analysis_acc = reduce_tensor(analysis_acc)

                loss_meters[j].update(analysis_loss.item(), analysis_target.size(0))
                acc_meters[j].update(analysis_acc.item(), analysis_target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info(
                    f'Test: [{idx}/{len(self.data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        # Update highest accuracies and log results
        for j in range(40):
            if acc_meters[j].avg > self.highest_acc_list[j]:
                self.highest_acc_list[j] = acc_meters[j].avg

        acc_list = [acc_meters[j].avg for j in range(40)]
        mean_acc = sum(acc_list) / len(acc_list)
        if mean_acc > self.highest_mean_acc:
            self.highest_mean_acc = mean_acc
            self.acc_list_corresponding_to_highest_mean_acc = acc_list.copy()

        self.limited_meter.append(mean_acc)

        if self.rank is 0:
            self.summary_writer: SummaryWriter

            for j in range(40):
                self.summary_writer.add_scalar(ANALYSIS_TASKS[j+1] + ' Val Loss', loss_meters[j].avg, global_step)

                logging.info('[%d]' % (global_step) + ANALYSIS_TASKS[j + 1] + ' Loss: %1.5f' % (loss_meters[j].avg))
                logging.info('[%d]' % (global_step) + ANALYSIS_TASKS[j + 1] + ' Accuracy: %1.5f' % (acc_meters[j].avg))
                logging.info('[%d]' % (global_step) + ANALYSIS_TASKS[j + 1] + ' Highest Accuracy: %1.5f' % (self.highest_acc_list[j]))

            logging.info('[%d]Mean Accuracy: %1.5f' % (global_step, mean_acc))
            logging.info('[%d]Max Mean Accuracy: %1.5f' % (global_step, self.highest_mean_acc))
            logging.info('[%d]10 Times Mean Accuracy: %1.5f' % (global_step, self.limited_meter.avg))
            logging.info('[%d]10 Times Max Mean Accuracy: %1.5f' % (global_step, self.limited_meter.best))

            temp = '[%d]Accs: ' % (global_step)
            for j in range(40):
                temp += "%1.5f " % self.acc_list_corresponding_to_highest_mean_acc[j]

            logging.info(temp)

    def __call__(self, num_update, model):
        """
        Calls the verification test function and ensures the model is in evaluation mode.
        """
        model.eval()
        self.ver_test(model, num_update)
        model.train()

# Similar classes for other datasets (FGNet, LAP, RAF) follow the same structure
# with specific metrics and logging for each dataset.

class FGNetVerification(object):
    """
    A class for verifying model performance on the FGNet dataset.
    Tracks mean absolute error (MAE) for age prediction.
    """
    def __init__(self, data_loader, summary_writer=None):
        """
        Initializes the FGNetVerification class with a data loader and an optional summary writer.
        """
        self.rank: int = distributed.get_rank()
        self.best_mae: float = 100.0

        self.data_loader = data_loader
        self.summary_writer = summary_writer
        self.limited_meter = LimitedAvgMeter(best_mode="min")

    def ver_test(self, model, global_step):
        """
        Performs verification testing on the model for the FGNet dataset.
        Computes and logs MAE for age prediction.
        """
        logging.info("Val on FGNet:")

        error_meter = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, target) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            age_result = model(img)

            error = torch.sum(torch.abs(age_result - target)) / target.size(0)

            error_meter.update(error.item(), target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info(
                    f'Test: [{idx}/{len(self.data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Error@1 {error_meter.val:.3f} ({error_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        if error_meter.avg < self.best_mae:
            self.best_mae = error_meter.avg

        self.limited_meter.append(error_meter.avg)

        if self.rank is 0:
            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag="age", scalar_value=error_meter.avg, global_step=global_step)
    
            logging.info('[%d]Mean Age Error: %1.5f' % (global_step, error_meter.avg))
            logging.info('[%d]MAE-Best: %1.5f' % (global_step, self.best_mae))
            logging.info('[%d]10 Times Mean Age Error: %1.5f' % (global_step, self.limited_meter.avg))
            logging.info('[%d]10 Times MAE-Best: %1.5f' % (global_step, self.limited_meter.best))

    def __call__(self, num_update, model):
        """
        Calls the verification test function and ensures the model is in evaluation mode.
        """
        model.eval()
        self.ver_test(model, num_update)
        model.train()

class LAPVerification(object):
    """
    A class for verifying model performance on the LAP dataset.
    Tracks mean absolute error (MAE) and E-error for age prediction.
    """
    def __init__(self, data_loader, summary_writer=None):
        """
        Initializes the LAPVerification class with a data loader and an optional summary writer.
        """
        self.rank: int = distributed.get_rank()
        self.best_mae: float = 100.0
        self.best_E_error: float = 100.0

        self.data_loader = data_loader
        self.summary_writer = summary_writer
        self.limited_meter = LimitedAvgMeter(best_mode="min")

    def ver_test(self, model, global_step):
        """
        Performs verification testing on the model for the LAP dataset.
        Computes and logs MAE and E-error for age prediction.
        """
        logging.info("Val on LAP:")

        mae_meter = AverageMeter()
        E_error_meter = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, target) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)
            mean = target[0].cuda(non_blocking=True)
            std = target[1].cuda(non_blocking=True)

            # Compute model output
            age_result = model(img)

            mae = torch.sum(torch.abs(age_result - mean)) / mean.size(0)
            E_error = 1 - torch.sum(torch.exp(-(age_result-mean)**2/(2*(std**2))))/mean.size(0)

            mae_meter.update(mae.item(), mean.size(0))
            E_error_meter.update(E_error.item(), mean.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        if mae_meter.avg < self.best_mae:
            self.best_mae = mae_meter.avg
        if E_error_meter.avg < self.best_E_error:
            self.best_E_error = E_error_meter.avg

        self.limited_meter.append(E_error_meter.avg)

        if self.rank is 0:
            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag="mae", scalar_value=mae_meter.avg, global_step=global_step)
            self.summary_writer.add_scalar(tag="E error", scalar_value=E_error_meter.avg, global_step=global_step)

            logging.info('[%d]Mean Age Error: %1.5f' % (global_step, mae_meter.avg))
            logging.info('[%d]MAE-Best: %1.5f' % (global_step, self.best_mae))
            logging.info('[%d]E Error: %1.5f' % (global_step, E_error_meter.avg))
            logging.info('[%d]E Error-Best: %1.5f' % (global_step, self.best_E_error))
            logging.info('[%d]10 Times E Error: %1.5f' % (global_step, self.limited_meter.avg))
            logging.info('[%d]10 Times E Error-Best: %1.5f' % (global_step, self.limited_meter.best))

    def __call__(self, num_update, model):
        """
        Calls the verification test function and ensures the model is in evaluation mode.
        """
        model.eval()
        self.ver_test(model, num_update)
        model.train()

class RAFVerification(object):
    """
    A class for verifying model performance on the RAF dataset.
    Tracks accuracy for emotion recognition.
    """
    def __init__(self, data_loader, summary_writer=None):
        """
        Initializes the RAFVerification class with a data loader and an optional summary writer.
        """
        self.rank: int = distributed.get_rank()
        self.highest_acc1: float = 0.0
        self.highest_acc5: float = 0.0

        self.data_loader = data_loader
        self.summary_writer = summary_writer
        self.limited_meter = LimitedAvgMeter(best_mode="max")

    def ver_test(self, model, global_step):
        """
        Performs verification testing on the model for the RAF dataset.
        Computes and logs accuracy for emotion recognition.
        """
        logging.info("Val on RAF:")

        criterion = torch.nn.CrossEntropyLoss()

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, target) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # Compute model output
            expression_result = model(img)

            loss = criterion(expression_result, target)
            acc1, acc5 = accuracy(expression_result, target, topk=(1, 5))

            loss = reduce_tensor(loss)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info(
                    f'Test: [{idx}/{len(self.data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        if acc1_meter.avg > self.highest_acc1:
            self.highest_acc1 = acc1_meter.avg
        if acc5_meter.avg > self.highest_acc5:
            self.highest_acc5 = acc5_meter.avg

        self.limited_meter.append(acc1_meter.avg)
        
        if self.rank is 0:
            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag="expression loss", scalar_value=loss_meter.avg, global_step=global_step)
            self.summary_writer.add_scalar(tag="expression acc1", scalar_value=acc1_meter.avg, global_step=global_step)
            self.summary_writer.add_scalar(tag="expression acc5", scalar_value=acc5_meter.avg, global_step=global_step)
    
            logging.info('[%d]Expression Loss: %1.5f' % (global_step, loss_meter.avg))
            logging.info('[%d]Expression Acc@1: %1.5f' % (global_step, acc1_meter.avg))
            logging.info('[%d]Expression Acc@1-Highest: %1.5f' % (global_step, self.highest_acc1))
            logging.info('[%d]Expression Acc@5: %1.5f' % (global_step, acc5_meter.avg))
            logging.info('[%d]Expression Acc@5-Highest: %1.5f' % (global_step, self.highest_acc5))
            logging.info('[%d]10 Times Expression Acc@1: %1.5f' % (global_step, self.limited_meter.avg))
            logging.info('[%d]10 Times Expression Acc@1-Highest: %1.5f' % (global_step, self.limited_meter.best))

    def __call__(self, num_update, model):
        """
        Calls the verification test function and ensures the model is in evaluation mode.
        """
        model.eval()
        self.ver_test(model, num_update)
        model.train()