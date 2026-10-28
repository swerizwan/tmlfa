import argparse
import logging
import os
from itertools import cycle
import math

from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from data import get_dataloader
from lossfuntions import CombinedMarginLoss
from scheduler import build_scheduler
from layer import PartialFC, PartialFCAdamW
from analysis import *
from analysis import subnets
from model import build_model

from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

# Ensure the PyTorch version is at least 1.9.0
assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    # Initialize distributed training environment
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    # Default values for single-process training
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    """
    Main function to start distributed training.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Get configuration from the provided config file
    cfg = get_config(args.config)
    # Set a global seed for reproducibility
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    # Set the CUDA device for the current process
    torch.cuda.set_device(args.local_rank)

    # Create the result directory if it doesn't exist
    os.makedirs(cfg.result, exist_ok=True)
    # Initialize logging
    init_logging(rank, cfg.result)

    # Create a TensorBoard summary writer for logging
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.result, "tensorboard"))
        if rank == 0
        else None
    )

    # Create data loaders for different tasks
    train_loader = get_analysis_train_dataloader("recognition", cfg, args.local_rank)
    age_gender_train_loader = get_analysis_train_dataloader("age_gender", cfg, args.local_rank)
    CelebA_train_loader = get_analysis_train_dataloader("CelebA", cfg, args.local_rank)
    Expression_train_loader = get_analysis_train_dataloader("expression", cfg, args.local_rank)

    # Build the model and move it to the GPU
    model = build_model(cfg).cuda()

    # Wrap the model with DistributedDataParallel for distributed training
    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()

    # Set the model to static graph mode
    model._set_static_graph()

    # Calculate total batch size and epoch steps
    cfg.total_batch_size = world_size * (cfg.recognition_bz + cfg.age_gender_bz + cfg.CelebA_bz + cfg.expression_bz)
    cfg.epoch_step = len(train_loader)

    # Calculate the number of epochs based on total steps
    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    # Adjust learning rates based on total batch size
    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Define loss functions for different tasks
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    age_loss = AgeLoss(total_iter=cfg.total_step)

    criteria = [age_loss]
    criteria.extend([torch.nn.CrossEntropyLoss() for j in range(41)])  # Total:42

    # Define optimizer and partial FC layer
    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    # Build the learning rate scheduler
    lr_scheduler = build_scheduler(
        optimizer=opt,
        lr_name=cfg.lr_name,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        num_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step)

    # Initialize training state variables
    start_epoch = 0
    global_step = 0

    # Load initial model if specified
    if cfg.init:
        dict_checkpoint = torch.load(os.path.join(cfg.init_model, f"start_{rank}.pt"))
        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"], strict=False)
        del dict_checkpoint

    # Load checkpoint if resuming training
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.result, f"checkpoint_step_{cfg.resume_step}_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        local_step = dict_checkpoint["local_step"]

        if local_step == cfg.epoch_step - 1:
            start_epoch += 1
            local_step = 0
        else:
            local_step += 1

        global_step += 1

        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        model.module.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
        model.module.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
        model.module.om.load_state_dict(dict_checkpoint["state_dict_om"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    # Log configuration details
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # Define callbacks for verification and logging
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )

    FGNet_loader = get_analysis_val_dataloader(data_choose="FGNet", config=cfg)
    LAP_loader = get_analysis_val_dataloader(data_choose="LAP", config=cfg)
    CelebA_loader = get_analysis_val_dataloader(data_choose="CelebA", config=cfg)
    RAF_loader = get_analysis_val_dataloader(data_choose="RAF", config=cfg)

    FGNet_verification = FGNetVerification(data_loader=FGNet_loader, summary_writer=summary_writer)
    LAP_verification = LAPVerification(data_loader=LAP_loader, summary_writer=summary_writer)
    CelebA_verification = CelebAVerification(data_loader=CelebA_loader, summary_writer=summary_writer)
    RAF_verification = RAFVerification(data_loader=RAF_loader, summary_writer=summary_writer)

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    # Initialize loss meters
    loss_am = AverageMeter()
    recognition_loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for j in range(42)]

    # Initialize AMP scaler for mixed precision training
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # Define batch sizes for different tasks
    bzs = [cfg.recognition_bz, cfg.age_gender_bz, cfg.CelebA_bz, cfg.expression_bz]

    # Calculate feature cuts for different tasks
    features_cut = [0 for i in range(5)]
    for i in range(1, 5):
        features_cut[i] = features_cut[i - 1] + bzs[i - 1]

    # Training loop
    for epoch in range(start_epoch, cfg.num_epoch):
        # Set epoch for distributed samplers
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(age_gender_train_loader, DataLoader):
            age_gender_train_loader.sampler.set_epoch(epoch)
        if isinstance(CelebA_train_loader, DataLoader):
            CelebA_train_loader.sampler.set_epoch(epoch)
        if isinstance(Expression_train_loader, DataLoader):
            Expression_train_loader.sampler.set_epoch(epoch)

        # Iterate over the data loaders
        for idx, data in enumerate(
                zip(train_loader, age_gender_train_loader, CelebA_train_loader, Expression_train_loader)):

            # Skip steps if resuming training
            if cfg.resume:
                if idx < local_step:
                    continue

            # Unpack data from different tasks
            recognition = data[0]
            age_gender = data[1]
            CelebA = data[2]
            RAF = data[3]

            recognition_img, recognition_label = recognition
            age_gender_img, [age_label, gender_label_1] = age_gender
            CelebA_img, CelebA_label = CelebA
            expression_img, expression_label = RAF
            gender_label = torch.cat([gender_label_1, CelebA_label[4]])

            # Move labels to GPU
            recognition_label = recognition_label.cuda(non_blocking=True)
            age_label = age_label.cuda(non_blocking=True)
            gender_label = gender_label.cuda(non_blocking=True)
            expression_label = expression_label.cuda(non_blocking=True)

            # Prepare analysis labels
            analysis_labels = [age_label]
            for j in range(40):
                analysis_labels.append(CelebA_label[j].cuda(non_blocking=True))
            analysis_labels[5] = gender_label
            analysis_labels.append(expression_label)

            # Concatenate images from different tasks
            img = torch.cat([recognition_img, age_gender_img, CelebA_img, expression_img], dim=0).cuda(
                non_blocking=True)

            # Set model to list mode for multi-task training
            model.module.set_result_type("List")
            results = model(img)

            # Calculate recognition loss
            local_embeddings = results[-1][features_cut[0]: features_cut[1]]
            recognition_loss = module_partial_fc(local_embeddings, recognition_label, opt)

            # Calculate analysis losses
            analysis_losses = []
            for j in range(42):
                if j == 0:  # age
                    analysis_result = results[j][features_cut[1]: features_cut[2]]
                    analysis_loss = criteria[j](analysis_result, analysis_labels[j], global_step)
                elif j == 5:
                    analysis_result = results[j][features_cut[1]: features_cut[3]]
                    analysis_loss = criteria[j](analysis_result, analysis_labels[j])
                elif j == 41:
                    analysis_result = results[j][features_cut[3]: features_cut[4]]
                    analysis_loss = criteria[j](analysis_result, analysis_labels[j])
                else:
                    analysis_result = results[j][features_cut[2]: features_cut[3]]
                    analysis_loss = criteria[j](analysis_result, analysis_labels[j])

                analysis_losses.append(analysis_loss)

            # Combine losses
            loss = cfg.recognition_loss_weight * recognition_loss
            for j in range(42):
                loss += analysis_losses[j] * cfg.analysis_loss_weights[j]

            # Backward pass and optimization
            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                opt.step()

            # Zero gradients and update learning rate
            opt.zero_grad()
            lr_scheduler.step_update(global_step)

            # Update loss meters
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                recognition_loss_am.update(recognition_loss.item(), 1)
                for j in range(42):
                    analysis_loss_ams[j].update(analysis_losses[j].item(), 1)

                # Log training progress
                callback_logging(global_step, loss_am, recognition_loss_am, analysis_loss_ams, epoch, cfg.fp16,
                                 lr_scheduler.get_update_values(global_step)[0], amp)

                # Perform verification at specified intervals
                if (global_step + 1) % cfg.verbose == 0:
                    model.module.set_result_type("Recognition")
                    callback_verification(global_step, model)
                    model.module.set_result_type("Age")
                    FGNet_verification(global_step, model)
                    LAP_verification(global_step, model)
                    model.module.set_result_type("Attribute")
                    CelebA_verification(global_step, model)
                    model.module.set_result_type("Expression")
                    RAF_verification(global_step, model)

            # Save model state at specified intervals
            if cfg.save_all_states and (global_step + 1) % cfg.save_verbose == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "local_step": idx,

                    "state_dict_backbone": model.module.backbone.state_dict(),
                    "state_dict_softmax_fc": module_partial_fc.state_dict(),
                    "state_dict_fam": model.module.fam.state_dict(),
                    "state_dict_tss": model.module.tss.state_dict(),
                    "state_dict_om": model.module.om.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(cfg.result, f"checkpoint_step_{global_step}_gpu_{rank}.pt"))

            # Check if training is complete
            if global_step >= cfg.total_step - 1:
                break  # end
            else:
                global_step += 1

        # Check if training is complete
        if global_step >= cfg.total_step - 1:
            break
        if cfg.dali:
            train_loader.reset()

    # Final verification
    with torch.no_grad():
        model.module.set_result_type("Recognition")
        callback_verification(global_step, model)
        model.module.set_result_type("Age")
        FGNet_verification(global_step, model)
        LAP_verification(global_step, model)
        model.module.set_result_type("Attribute")
        CelebA_verification(global_step, model)
        model.module.set_result_type("Expression")
        RAF_verification(global_step, model)

    # Clean up distributed training
    distributed.destroy_process_group()


if __name__ == "__main__":
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())