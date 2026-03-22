import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import numpy as np
from pathlib import Path
import yaml
import pprint
from dotmap import DotMap
import datetime
import shutil
from contextlib import suppress

from utils.utils import epoch_saving, best_saving, AverageMeter
from utils.logger import setup_logger
from utils.Augmentation import get_augmentation
from utils.solver import _optimizer, _lr_scheduler
from modules.video_clip_mergev5_mae import VidPrism


# ===== transformers for VideoMAEv2 =====
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig


def update_dict(d):
    """Remove the `module.` prefix added by DDP."""
    new_dict = {}
    for k, v in d.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict



def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        print('[WARN] Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    dist_backend = args.dist_backend
    dist.init_process_group(backend=dist_backend, init_method='env://')
    dist.barrier()


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--precision', choices=['amp', 'fp16', 'fp32'], default='amp')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--local_rank', default=None, type=int)
    args = parser.parse_args()
    return args


class VideoMAEFeatureExtractor(nn.Module):
    def __init__(self, videomae_model, num_segments):
        super().__init__()
        self.videomae = videomae_model
        self.num_segments = num_segments
        proj = self.videomae.model.patch_embed.proj
        self.t_patch, self.h_patch, self.w_patch = proj.kernel_size

    def forward(self, video_tensor):
        b_times_t, c, h, w = video_tensor.shape
        t = self.num_segments
        b = b_times_t // t
        frames = video_tensor.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        patch_tokens = self.videomae(frames)  # Output can be [B, N, D] or [B, T_patches, D]

        # Infer the output shape automatically
        if patch_tokens.ndim == 3:
            B, N, D = patch_tokens.shape
        else:
            raise RuntimeError(f"Unexpected VideoMAEv2 output shape: {patch_tokens.shape}")

        # Assume N == T_patches * H_patches * W_patches
        T_patches = frames.shape[2] // self.t_patch
        H_patches = frames.shape[3] // self.h_patch
        W_patches = frames.shape[4] // self.w_patch
        expected_tokens = T_patches * H_patches * W_patches

        if N == expected_tokens:
            patch_tokens = patch_tokens.view(B, T_patches, H_patches * W_patches, D)
            time_tokens = patch_tokens.mean(2)
        elif N == T_patches:
            time_tokens = patch_tokens
        else:
            # fail-safe fallback
            print(f"[WARN] Unrecognized patch layout: N={N}, expected {expected_tokens} or {T_patches}")
            time_tokens = patch_tokens.mean(1, keepdim=True)

        return time_tokens  # [B, T, D]

    def encode_image(self, video_tensor):
        return self.forward(video_tensor)


def main(args):
    global best_prec1
    init_distributed_mode(args)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)

    # Working dir
    working_dir = os.path.join(config['data']['output_path'], config['data']['dataset'], "VideoMAEv2", args.log_time)
    if is_main_process():
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(args.config, working_dir)
            shutil.copy(Path(__file__).name, working_dir)
        except Exception:
            pass
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    logger = setup_logger(output=working_dir, distributed_rank=(dist.get_rank() if dist.is_initialized() else 0), name='VideoMAEv2')
    if is_main_process():
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"TorchVision: {torchvision.__version__}")
        logger.info(pprint.pformat(config.toDict() if hasattr(config, 'toDict') else dict(config)))


    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(args.local_rank or 0)
        cudnn.benchmark = True

    # Seed
    seed = int(config.seed)
    if dist.is_available() and dist.is_initialized():
        seed += dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use VideoMAEv2 as the visual backbone
    if config.network.arch == 'ViT-B/16':
        video_config = AutoConfig.from_pretrained("/home/linrui/moe/VideoMAEv2-Base", trust_remote_code=True)
        video_model = AutoModel.from_pretrained('/home/linrui/moe/VideoMAEv2-Base', config=video_config, trust_remote_code=True)
    else:
        video_config = AutoConfig.from_pretrained("/home/linrui/moe/VideoMAEv2-Large", trust_remote_code=True)
        video_model = AutoModel.from_pretrained('/home/linrui/moe/VideoMAEv2-Large', config=video_config, trust_remote_code=True)
    if args.precision in ("amp", "fp32"):
        video_model = video_model.float()
    # print(video_model)
    d_model = video_model.model.embed_dim
    print(f"[INFO] VideoMAEv2 model loaded. d_model={d_model}")
    model = VideoMAEFeatureExtractor(video_model, config.data.num_segments)

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    # Dataset
    from datasets.video import Video_dataset
    train_data = Video_dataset(
        config.data.train_root, config.data.train_list,
        config.data.label_list, num_segments=config.data.num_segments,
        modality=config.data.modality, image_tmpl=config.data.image_tmpl,
        random_shift=config.data.random_shift,
        transform=transform_train, dense_sample=config.data.dense)
    val_data = Video_dataset(
        config.data.val_root, config.data.val_list,
        config.data.label_list, num_segments=config.data.num_segments,
        modality=config.data.modality, image_tmpl=config.data.image_tmpl,
        transform=transform_val, dense_sample=config.data.dense)

    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_data, shuffle=True)
        val_sampler = DistributedSampler(val_data, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_data, batch_size=config.data.batch_size,
                              num_workers=config.data.workers, sampler=train_sampler,
                              shuffle=(train_sampler is None), drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size,
                            num_workers=config.data.workers, sampler=val_sampler,
                            shuffle=False, drop_last=False, pin_memory=True)

    # Loss: cross-entropy classification only
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if args.precision == 'fp16':
        criterion = criterion.half()

    # MoE head
    video_head = VidPrism(
        num_experts=config.network.num_experts,
        sampling_rates=config.network.sampling_rates,
        num_classes=config.data.num_classes,
        d_model=d_model,
    )

    start_epoch = int(config.solver.start_epoch)

    # Load pretrained weights or resume training
    if config.pretrain and os.path.isfile(config.pretrain):
        if is_main_process():
            logger.info(f"=> Transfer learning from {config.pretrain}")
        checkpoint = torch.load(config.pretrain, map_location='cpu')
        model.load_state_dict(update_dict(checkpoint['model_state_dict']), strict=True)

        checkpoint_head_state_dict = update_dict(checkpoint['fusion_model_state_dict'])
        current_head_state_dict = video_head.state_dict()
        new_state_to_load = {k: v for k, v in checkpoint_head_state_dict.items()
                             if k in current_head_state_dict and v.shape == current_head_state_dict[k].shape}
        video_head.load_state_dict(new_state_to_load, strict=False)
    elif config.resume and os.path.isfile(config.resume):
        if is_main_process():
            logger.info(f"=> Resume training from {config.resume}")
        checkpoint = torch.load(config.resume, map_location='cpu')
        model.load_state_dict(update_dict(checkpoint['model_state_dict']))
        video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1

    # Freeze the video backbone
    if config.network.fix_video:
        for param in model.parameters():
            param.requires_grad_(False)

    model.to(device)
    video_head.to(device)

    if dist.is_available() and dist.is_initialized():
        trainable = [n for n,p in model.named_parameters() if p.requires_grad]
        trainable_videohead = [n for n,p in video_head.named_parameters() if p.requires_grad]
        print(f"[DEBUG] model trainable param count: {len(trainable)}")      
        print(f"[DEBUG] video_head trainable param count: {len(trainable_videohead)}")  
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        video_head = DDP(video_head, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    optimizer = _optimizer(config, model, video_head)
    lr_scheduler = _lr_scheduler(config, optimizer)
    scaler = GradScaler(enabled=(args.precision == 'amp'))

    best_prec1 = 0.0

    if config.solver.evaluate:
        validate(start_epoch, val_loader, device, model, video_head, config, logger, args)
        cleanup_ddp()
        return

    for epoch in range(start_epoch, config.solver.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train(model, video_head, train_loader, optimizer, criterion, scaler, epoch, device, lr_scheduler, config, logger, args)

        if (epoch + 1) % config.logging.eval_freq == 0:
            prec1, prec5 = validate(epoch, val_loader, device, model, video_head, config, logger, args)
            if is_main_process():
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1, best_prec1))
                logger.info('Saving:')
                filename = f"{working_dir}/last_model.pt"
                model_to_save = model.module if isinstance(model, DDP) else model
                head_to_save = video_head.module if isinstance(video_head, DDP) else video_head
                epoch_saving(epoch, model_to_save, head_to_save, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model_to_save, head_to_save, optimizer)

    cleanup_ddp()


def train(model, video_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, logger, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    video_head.train()
    autocast_ctx = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()

    for i, (images, list_id) in enumerate(train_loader):
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))

        list_id = list_id.to(device, non_blocking=True)
        data_time.update(time.time() - end)
        images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])
        b, _, c, h, w = images.size()
        images = images.view(-1, c, h, w).to(device, non_blocking=True)

        with autocast_ctx():
            images = model(images)
            _, t, _ = images.shape
            # print(f"[DEBUG] images shape after model: {images.shape}")  # [B, T, D]
            logits_exp, gating_weights, div_loss, rank_loss = video_head(images, b, t)
            loss_exp = criterion(logits_exp, list_id)

            gate_loss = 0.0
            if gating_weights is not None:
                num_experts = gating_weights.shape[-1]
                importance_per_expert = gating_weights.mean(dim=0)
                load_per_expert = gating_weights.sum(dim=0) / gating_weights.shape[0]
                gate_loss = num_experts * torch.sum(importance_per_expert * load_per_expert)

            loss = loss_exp + 0.01 * gate_loss + 0.005 * div_loss + 0.1 * rank_loss
            loss = loss / config.solver.grad_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        losses.update(loss.item() * config.solver.grad_accumulation_steps, logits_exp.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        cur_iter = epoch * len(train_loader) + i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

        if is_main_process() and (i % config.logging.print_freq == 0):
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'loss_exp {loss_exp:.4f} g_loss {gate_loss:.4f} d_loss {div_loss:.4f} rank_loss {rank_loss:.4f}').format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             loss_exp=loss_exp.item(), gate_loss=gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
                             div_loss=div_loss.item(), rank_loss=rank_loss,
                             lr=optimizer.param_groups[-1]['lr']))


def validate(epoch, val_loader, device, model, video_head, config, logger, args):
    model.eval()
    video_head.eval()
    autocast_ctx = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    total_correct1 = 0.0
    total_correct5 = 0.0
    total_count = 0

    with torch.no_grad():
        for i, (image, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, _, c, h, w = image.size()
            class_id = class_id.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True).view(-1, c, h, w)
            with autocast_ctx():
                base_model = model.module if isinstance(model, DDP) else model
                image_embedding = base_model.encode_image(image)
                _, t, _ = image_embedding.shape
                similarity, _, _, _ = video_head(image_embedding, b, t)

            maxk = max((1, 5))
            _, pred = similarity.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(class_id.view(1, -1).expand_as(pred))
            correct1 = correct[:1].reshape(-1).float().sum().item()
            correct5 = correct[:5].reshape(-1).float().sum().item()
            bs = class_id.size(0)
            total_correct1 += correct1
            total_correct5 += correct5
            total_count += bs
            if is_main_process() and i % config.logging.print_freq == 0:
                # Local (pre-reduce) snapshot
                local_top1 = 100.0 * correct1 / bs
                local_top5 = 100.0 * correct5 / bs
                logger.info((
                    'Test: [{0}/{1}]\tPrec@1 {top1:.3f}\tPrec@5 {top5:.3f}'
                ).format(i, len(val_loader), top1=local_top1, top5=local_top5))

    if dist.is_available() and dist.is_initialized():
        t1 = torch.tensor([total_correct1], device=device)
        t5 = torch.tensor([total_correct5], device=device)
        tc = torch.tensor([total_count], device=device)
        dist.all_reduce(t1, op=dist.ReduceOp.SUM)
        dist.all_reduce(t5, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
        total_correct1 = t1.item()
        total_correct5 = t5.item()
        total_count = int(tc.item())

    top1_avg = 100.0 * total_correct1 / max(1, total_count)
    top5_avg = 100.0 * total_correct5 / max(1, total_count)

    if is_main_process():
        logger.info(f"Testing Results: Prec@1 {top1_avg:.3f} Prec@5 {top5_avg:.3f}")

    return top1_avg, top5_avg


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_parser()
    main(args)
