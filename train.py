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
from torch.utils import checkpoint as _cp
_orig_cp = _cp.checkpoint

def _checkpoint_nonreentrant(function, *args, **kwargs):
    kwargs["use_reentrant"] = False
    return _orig_cp(function, *args, **kwargs)

_cp.checkpoint = _checkpoint_nonreentrant

from utils.utils import epoch_saving, best_saving, AverageMeter, accuracy
from utils.logger import setup_logger
import clip

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap

import datetime
import shutil
from contextlib import suppress

from modules.video_clip_mergev5 import VidPrism, VideoCLIP
from utils.Augmentation import get_augmentation
from utils.solver import _optimizer, _lr_scheduler, _lr_scheduler_step
from utils.utils import init_distributed_mode
from modules.text_prompt import text_prompt, text_prompt_ensemble, text_prompt_ensemble_for_ssv2



def update_dict(d):
    new_dict = {}
    for k, v in d.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict



def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')

    # Precision
    parser.add_argument('--precision', choices=['amp', 'fp16', 'fp32'], default='amp', help='Floating point precision.')

    # DDP-specific
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--local_rank', default=None, type=int, help='torchrun sets LOCAL_RANK automatically')

    args = parser.parse_args()
    return args


def main(args):
    global best_prec1

    init_distributed_mode(args)

    # Load config (all ranks)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)

    # Working directory (created by rank0)
    working_dir = os.path.join(config['data']['output_path'], config['data']['dataset'], config['network']['arch'], args.log_time)
    if is_main_process():
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(args.config, working_dir)
            shutil.copy(Path(__file__).name, working_dir)
        except Exception:
            pass
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Logger
    logger = setup_logger(output=working_dir, distributed_rank=(dist.get_rank() if dist.is_initialized() else 0), name='SFMoE')
    if is_main_process():
        logger.info("------------------------------------")
        logger.info("Environment Versions:")
        logger.info(f"- Python: {sys.version}")
        logger.info(f"- PyTorch: {torch.__version__}")
        logger.info(f"- TorchVision: {torchvision.__version__}")
        logger.info("------------------------------------")
        pp = pprint.PrettyPrinter(indent=4)
        logger.info(pp.pformat(config.toDict() if hasattr(config, 'toDict') else dict(config)))
        logger.info("------------------------------------")
        logger.info(f"storing name: {working_dir}")


    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(args.local_rank or 0)
        cudnn.benchmark = True

    # Seed (rank-dependent to avoid identical shuffles if needed)
    seed = int(config.seed)
    if dist.is_available() and dist.is_initialized():
        seed = seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # CLIP backbone
    if config.network.arch.startswith('16shot'):
        arch = 'ViT-L/14'
    else:
        arch = config.network.arch

    model_clip, _ = clip.load(
        arch,
        device='cpu', jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st=config.network.joint_st)

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)
    if is_main_process():
        logger.info('train transforms: {}'.format(transform_train.transforms))
        logger.info('val transforms: {}'.format(transform_val.transforms))

    if args.precision in ("amp", "fp32"):
        model_clip = model_clip.float()

    # Datasets
    if config.data.dataset == 'charades':
        from datasets.charades import Video_dataset
        # (Assumed unchanged)
    else:
        from datasets.video import Video_dataset, Video_dataset_few
        train_data = Video_dataset_few(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train, dense_sample=config.data.dense)
        val_data = Video_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, dense_sample=config.data.dense)

    # Samplers & Loaders
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_data, shuffle=True)
        val_sampler = DistributedSampler(val_data, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # Loss
    loss_type = config.solver.loss_type
    if loss_type == 'CE':
        criterion = torch.nn.CrossEntropyLoss().to(device)
        if args.precision == 'fp16':
            criterion = criterion.half()
    else:
        raise NotImplementedError

    # Encode class text features (rank0 does it; broadcast to others)
    if is_main_process():
        print('============= Start encoding class features ============')
    if config.data.dataset == 'ssv2':
        classes = text_prompt_ensemble_for_ssv2(train_data)
    else:
        classes = text_prompt_ensemble(train_data)
    model_clip.to(device)
    model_clip.eval()
    d_model = model_clip.visual.output_dim
    # print(f"[Notice]Text encoder output dim: {d_model}")
    if hasattr(model_clip, "gradient_checkpointing_enable"):
        model_clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with torch.no_grad():
        cls_feature_list = [model_clip.encode_text(classes[i].to(device), return_token=True)[0] for i in range(len(classes))]
        for cf in cls_feature_list:
            cf /= cf.norm(dim=-1, keepdim=True)
        cls_feature = torch.stack(cls_feature_list, 0).mean(0)
        cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
    if is_main_process():
        print('============= End encoding class features ============')

    # Build models
    model = VideoCLIP(model_clip, config.data.num_segments)
    del model_clip

    video_head = VidPrism(
        num_experts=config.network.num_experts,
        sampling_rates=config.network.sampling_rates,
        num_classes=config.data.num_classes,
        d_model=d_model,
    )

    start_epoch = int(config.solver.start_epoch)

# Load checkpoint for transfer learning (rank0 loads then broadcast)
    if config.pretrain: # Use config.pretrain to trigger transfer learning
        ckpt_path = config.pretrain
        if is_main_process() and os.path.isfile(ckpt_path):
            logger.info("=============================================")
            logger.info(f"=> Starting transfer learning from: '{ckpt_path}'")
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            model.load_state_dict(update_dict(checkpoint['model_state_dict']), strict=True)
            logger.info("Successfully loaded backbone model weights.")

            checkpoint_head_state_dict = update_dict(checkpoint['fusion_model_state_dict'])
            current_head_state_dict = video_head.state_dict()
            
            # Build a new dict with only weights present in the current model and matching in shape
            new_state_to_load = {}
            for k, v_pretrained in checkpoint_head_state_dict.items():
                if k in current_head_state_dict and v_pretrained.shape == current_head_state_dict[k].shape:
                    new_state_to_load[k] = v_pretrained
                else:
                    logger.warning(f"-> Skipping layer '{k}' in video_head due to shape mismatch "
                                   f"(checkpoint shape: {v_pretrained.shape}, "
                                   f"current model shape: {current_head_state_dict.get(k, 'N/A')}). "
                                   "This is expected for the classifier head in transfer learning.")
            
            # Load the filtered weights with strict=False so missing classifier weights do not raise errors
            video_head.load_state_dict(new_state_to_load, strict=False)
            logger.info("Successfully loaded compatible weights for video_head. Classifier head is randomly initialized.")
            logger.info("=============================================")
            
            del checkpoint

    elif config.resume:
        ckpt_path = config.resume
        if is_main_process() and os.path.isfile(ckpt_path):
            logger.info(f"=> Resuming from checkpoint '{ckpt_path}'")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(update_dict(checkpoint['model_state_dict']))
            video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
            if 'epoch' in checkpoint:
                start_epoch = int(checkpoint['epoch']) + 1
                logger.info(f"=> Resumed from epoch {checkpoint['epoch']}")
            del checkpoint

        model.to(device)
        video_head.to(device)
        if dist.is_available() and dist.is_initialized():
            # Broadcast weights from rank0
            for p in model.parameters():
                dist.broadcast(p.data, src=0)
            for p in video_head.parameters():
                dist.broadcast(p.data, src=0)
            # Broadcast buffers as well
            for b in model.buffers():
                dist.broadcast(b.data, src=0)
            for b in video_head.buffers():
                dist.broadcast(b.data, src=0)
            # Broadcast start_epoch
            start_epoch_tensor = torch.tensor([start_epoch], device=device)
            dist.broadcast(start_epoch_tensor, src=0)
            start_epoch = int(start_epoch_tensor.item())

    # Freeze video backbone if needed
    if config.network.fix_video:
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)

    if config.network.fix_text:
        for name, param in model.named_parameters():
            if "transformer" in name or "token_embedding" in name or "positional_embedding" in name:
                param.requires_grad_(False)

    # Move to device
    model.to(device)
    video_head.to(device)

    # DDP wrap
    if dist.is_available() and dist.is_initialized():
        # trainable = [n for n,p in model.named_parameters() if p.requires_grad]
        # print(f"[DEBUG] model trainable param count: {len(trainable)}")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        video_head = DDP(video_head, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Optimizer & LR scheduler
    optimizer = _optimizer(config, model, video_head)
    lr_scheduler = _lr_scheduler(config, optimizer)

    # AMP scaler per-rank
    scaler = GradScaler(enabled=(args.precision == 'amp'))

    best_prec1 = 0.0

    if config.solver.evaluate:
        if is_main_process():
            logger.info("=========== evaluate ===========")
        prec1, prec5 = validate(start_epoch, val_loader, device, model, video_head, config, cls_feature, logger, args)
        cleanup_ddp()
        return

    # Train
    for epoch in range(start_epoch, config.solver.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train(model, video_head, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, cls_feature, logger, args)

        # Validation on schedule (all ranks run; reduce metrics)
        if (epoch + 1) % config.logging.eval_freq == 0:
            prec1, prec5 = validate(epoch, val_loader, device, model, video_head, config, cls_feature, logger, args)

            if is_main_process():
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1, best_prec1))
                logger.info('Saving:')
                filename = f"{working_dir}/last_model.pt"

                # Save .module if wrapped by DDP
                model_to_save = model.module if isinstance(model, DDP) else model
                head_to_save = video_head.module if isinstance(video_head, DDP) else video_head
                epoch_saving(epoch, model_to_save, head_to_save, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model_to_save, head_to_save, optimizer)


    cleanup_ddp()


def train(model, video_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, text_embedding, logger, args):
    """Train one epoch on (possibly) multiple GPUs."""
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
        b, t, c, h, w = images.size()
        images = images.view(-1, c, h, w).to(device, non_blocking=True)

        with autocast_ctx():
            images = model(images)

            txt_b = text_embedding[list_id].to(device)          # [B, D]
            # Cosine similarity -> temporal prior [B, T]
            feats_bt = F.normalize(images, dim=-1)              # [B, T, D]
            txt_b = F.normalize(txt_b,   dim=-1)             # [B, D]
            prior = torch.einsum('btd,bd->bt', feats_bt, txt_b)
            # Smooth and apply softmax (temperature is tunable, e.g. 0.15-0.3)
            prior = F.avg_pool1d(prior.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            prior = F.softmax(prior / 0.2, dim=-1).detach()     # Use as the target distribution to stop gradient flow
            
            logits_exp, gating_weights, fused_output, div_loss, rank_loss = video_head(images, b, t)
            loss_exp = criterion(logits_exp, list_id)

            gate_loss = 0.0
            if gating_weights is not None:
                num_experts = gating_weights.shape[-1]
                importance_per_expert = gating_weights.mean(dim=0)
                load_per_expert = gating_weights.sum(dim=0) / gating_weights.shape[0]
                gate_loss = num_experts * torch.sum(importance_per_expert * load_per_expert)

            fused_output_norm = F.normalize(fused_output, p=2, dim=1)
            text_embedding_norm = F.normalize(text_embedding.to(device), p=2, dim=1)
            logits_per_video = fused_output_norm @ text_embedding_norm.t()
            loss_contrastive = criterion(logits_per_video, list_id)

            loss = loss_exp + 0.01 * gate_loss + 0.01 * loss_contrastive + 0.005 * div_loss + 0.1 * rank_loss
            loss = loss / config.solver.grad_accumulation_steps

        if scaler is not None and scaler.is_enabled():
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

        # only for logging; we can use local loss as proxy
        losses.update(loss.item() * config.solver.grad_accumulation_steps, logits_exp.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        cur_iter = epoch * len(train_loader) + i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

        if is_main_process() and (i % config.logging.print_freq == 0):
            try:
                lr_val = optimizer.param_groups[-1]['lr']
            except Exception:
                lr_val = None

            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'loss_exp {loss_exp:.4f} g_loss {gate_loss:.4f} c_loss {contr_loss:.4f} d_loss {div_loss:.4f} rank_loss {rank_loss:.4f}').format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             loss_exp=loss_exp.item(), gate_loss=gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
                             contr_loss=loss_contrastive.item(), div_loss=div_loss.item(), rank_loss=rank_loss,
                             lr=optimizer.param_groups[-1]['lr']))


def validate(epoch, val_loader, device, model, video_head, config, text_embedding, logger, args):
    model.eval()
    video_head.eval()
    autocast_ctx = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    # We'll aggregate metrics across ranks
    total_correct1 = 0.0
    total_correct5 = 0.0
    total_count = 0

    with torch.no_grad():
        for i, (image, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device, non_blocking=True)
            text_embedding = text_embedding.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True).view(-1, c, h, w)
            with autocast_ctx():
                # call encode_image on underlying module if wrapped by DDP
                base_model = model.module if isinstance(model, DDP) else model
                image_embedding = base_model.encode_image(image)
                similarity, _, _, _, _ = video_head(image_embedding, b, t)

            # accuracy() likely returns percentages; compute counts instead
            # Convert to topk predictions
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

    # Reduce across ranks
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
        logger.info((
            'Testing Results: Prec@1 {top1:.3f} Prec@5 {top5:.3f}'
        ).format(top1=top1_avg, top5=top5_avg))

    return top1_avg, top5_avg


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_parser()
    main(args)
