import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import time
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
import clip
import numpy as np

import yaml
from dotmap import DotMap
from datasets.video import Video_dataset
from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
# from modules.video_clip_mergev1 import VidPrism, VideoCLIP
# from modules.video_clip_mergev5 import VidPrism, VideoCLIP
from VidPrism.modules.videomoe_text import VidPrism, VideoCLIP
from modules.text_prompt import text_prompt, text_prompt_ensemble

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument('--test_crops', type=int, default=1)   
    parser.add_argument('--test_clips', type=int, default=1) 
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use dense sample for test as in Non-local I3D')
    parser.add_argument('--distributed', default=False, action="store_true",
                        help='whether to use distributed test')
    args = parser.parse_args()
    return args

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


def main(args):
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed test', flush=True)
        rank = dist.get_rank()
    else:
        print('[INFO] turn off distributed test', flush=True)
        
        rank = 0

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # get fp16 model and weight
    model_clip, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu', jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st= config.network.joint_st) # Must set jit=False for training  ViT-B/32

    if args.precision == "amp" or args.precision == "fp32":
        model_clip = model_clip.float()


    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    # rescale size
    scale_size = int(config.data.input_size)

    # crop size
    input_size = config.data.input_size

    # control the spatial crop
    if args.test_crops == 1: # one center crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops (upper left, upper right, lower right, lower left, center)
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 10: # 5 normal crops + 5 flipped crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
            )
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))

    train_data = Video_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]), 
        dense_sample=config.data.dense)

    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        test_mode=True,
        transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]),
        dense_sample=args.dense,
        test_clips=args.test_clips)

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    
    train_loader = DataLoader(train_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)


    # ============= generate class features ==============
    print('============= Start encoding class features ===========')
    classes = text_prompt_ensemble(val_data)
    n_class = classes[0].size(0)
    model_clip.cuda()
    model_clip.eval()
    d_model = model_clip.visual.output_dim
    with torch.no_grad():
        # @zmhh_h multi text prompts
        cls_feature_list = [model_clip.encode_text(classes[i].cuda(), return_token=True)[0] for i in range(len(classes))]
        for cls_feature in cls_feature_list:
            cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
        cls_feature = torch.stack(cls_feature_list, 0).mean(0)
        cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
    print('============= End encoding class features ===========')

    # CLIP image encoder
    model = VideoCLIP(model_clip, config.data.num_segments)
    del model_clip

    # Temporal aggregation module using the same architecture as train.py
    video_head = VidPrism(num_experts=config.network.num_experts,
                                sampling_rates=config.network.sampling_rates,
                                num_classes=config.data.num_classes,
                                d_model=d_model)

    # =============== patch clip weights with a ratio of alpha===================
    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location='cpu')
        checkpoint_patch = {}
        alpha = 0.99
        for k, v in checkpoint['model_state_dict'].items():
            if k in clip_state_dict.keys():
                checkpoint_patch[k]= alpha*v+(1-alpha)*clip_state_dict[k]
            else:
                print('unmatched parameters: ',k)
        if rank == 0:
            print('load model: epoch {}'.format(checkpoint['epoch']))
        # model.load_state_dict(update_dict(checkpoint['model_state_dict']))
        model.load_state_dict(checkpoint_patch)
        video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
        del checkpoint,checkpoint_patch

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
        video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu])
    else:
        model = model.cuda()
        video_head = video_head.cuda()

    prec1,_, classes = validate(
        train_loader, device,
        model, video_head, config, cls_feature, args.test_crops, args.test_clips, args.distributed, rank)
    
    prec_train = validate_train(
        train_loader, device,
        model, video_head, config, cls_feature, args.test_crops, args.test_clips, args.distributed, rank, classes)
    return



import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import traceback  # Used to catch visualization errors without interrupting validation

# ==============================================================
# 1. Visualization helpers (placed before validate)
# ==============================================================

def unnormalize_img(tensor, mean, std):
    """
    Convert a normalized tensor [C, H, W] back to a uint8 numpy image [H, W, 3].
    """
    tensor = tensor.clone().detach().cpu()
    
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def unnormalize_tensor_robust(tensor):
    """Helper that robustly normalizes a tensor to the range [0, 1]."""
    
    g_min = tensor.min()
    g_max = tensor.max()
    if g_max > g_min:
        return (tensor - g_min) / (g_max - g_min)
    return tensor

def unnormalize_img_robust(tensor):
    """
    Robust inverse normalization using min-max scaling to [0, 255].
    This avoids overly dark or overexposed outputs when mean/std do not match.
    """
    img = tensor.clone().detach().cpu().permute(1, 2, 0).numpy() # [H, W, C]
    
    
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    
    
    img = (img * 255).astype(np.uint8)
    return img


def run_full_spatial_visualization(model, video_head, image_input, config, device, save_name="val_full_spatial_vis.png"):
    """
    Visualize spatial saliency for all experts across all frames.
    """
    
    # image_input: [T, C, H, W]
    T = image_input.shape[0] 
    
    
    target_frames = image_input.clone().detach()
    target_frames.requires_grad = True
    
    
    mean = config.data.mean if hasattr(config.data, 'mean') else [0.48145466, 0.4578275, 0.40821073]
    std = config.data.std if hasattr(config.data, 'std') else [0.26862954, 0.26130258, 0.27577711]
    raw_images = [unnormalize_img(f, mean, std) for f in target_frames]

    
    encoder = model.module if hasattr(model, 'module') else model
    
    # [1, T, C, H, W] -> Encoder -> [B, T, D] or [T, D]
    features_flat, _ = encoder(target_frames) 
    
    
    if features_flat.dim() == 2:
        features_seq = features_flat.unsqueeze(1) 
    else:
        features_seq = features_flat.permute(1, 0, 2)
    
    
    head_mod = video_head.module if hasattr(video_head, 'module') else video_head
    rates = head_mod.sampling_rates # [2, 4, 8, 16]
    num_experts = len(rates)
    
    results = {}

    print(f"[VIS] Computing saliency for {num_experts} experts over {T} frames...")

    for i in range(num_experts):
        if target_frames.grad is not None:
            target_frames.grad.zero_()
            
        pooler = head_mod.pooling_layers[i]
        feat_in = features_seq.permute(1, 2, 0) # -> [1, D, T]
        
        # Pooling
        pooled_out, _ = pooler(feat_in)
        
        # Loss & Backward
        loss = pooled_out.norm()
        loss.backward(retain_graph=True)
        
        
        grads = target_frames.grad.data
        saliency = grads.abs().max(dim=1)[0].cpu().numpy() # [T, H, W]
        results[i] = saliency

    
    
    fig_w = max(20, T * 0.8) 
    fig_h = num_experts * 2.0
    
    fig, axes = plt.subplots(num_experts, T, figsize=(fig_w, fig_h))
    
    
    if num_experts == 1: axes = axes[np.newaxis, :]

    plt.subplots_adjust(wspace=0.0, hspace=0.05) 

    for row in range(num_experts):
        saliency_maps = results[row]
        rate = rates[row]
        
        
        
        axes[row, 0].set_ylabel(f"Expert {row}\n(Rate={rate})", fontsize=12, fontweight='bold')

        for col in range(T):
            ax = axes[row, col]
            
            
            sal = saliency_maps[col]
            img = raw_images[col]
            
            # Resize
            sal_resized = cv2.resize(sal, (img.shape[1], img.shape[0]))
            
            
            
            sal_smoothed = cv2.GaussianBlur(sal_resized, (51, 51), 0)
            
            # Normalize
            if sal_smoothed.max() > 0:
                sal_smoothed = (sal_smoothed - sal_smoothed.min()) / (sal_smoothed.max() - sal_smoothed.min())
            
            # Heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * sal_smoothed), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
            
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            
            
            if row == 0 and (col % 4 == 0 or col == T-1):
                ax.set_title(f"T{col}", fontsize=10)

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Full visualization saved to {save_path}")


def visualize_temporal_token_merging(model, video_head, image_input, config, device, save_name="vis_temporal_merging.png"):
    """
    Visualize temporal token merging in a ToMe-like style.
    Colors indicate which source frames are merged into each target or representative frame.
    """
    
    n_seg = config.data.num_segments
    
    if image_input.dim() == 5: image_input = image_input[0] # handle [1, T, C, H, W]
    target_frames = image_input[:n_seg].clone().detach().to(device)
    
    
    mean = config.data.mean if hasattr(config.data, 'mean') else [0.48145466, 0.4578275, 0.40821073]
    std = config.data.std if hasattr(config.data, 'std') else [0.26862954, 0.26130258, 0.27577711]
    raw_images = [unnormalize_img(f, mean, std) for f in target_frames]
    T = len(raw_images)

    
    encoder = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        features_flat, _ = encoder(target_frames) # [1, T, D] or [T, D]
        if features_flat.dim() == 2: features_seq = features_flat.unsqueeze(1) # [T, 1, D]
        else: features_seq = features_flat.permute(1, 0, 2) # [T, B, D]
    
    
    head_mod = video_head.module if hasattr(video_head, 'module') else video_head
    rates = head_mod.sampling_rates
    num_experts = len(rates)

    
    
    cmap = plt.get_cmap('tab20')
    
    fig, axes = plt.subplots(num_experts, T, figsize=(max(20, T*0.8), num_experts*2.2))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    if num_experts == 1: axes = axes[np.newaxis, :]

    print(f"[VIS] Generating Merging Visualization for {num_experts} experts...")

    for exp_idx in range(num_experts):
        pooler = head_mod.pooling_layers[exp_idx]
        rate = rates[exp_idx]
        chunk_feat = features_seq.permute(1, 2, 0) # [1, D, T]
        
        
        # state: 0=Merged, 1=Representative
        
        frame_states = np.zeros(T, dtype=int) 
        frame_group_ids = np.zeros(T, dtype=int) - 1 

        
        
        for start in range(0, T, rate):
            end = min(start + rate, T)
            chunk = chunk_feat[:, :, start:end] # [1, D, wlen]
            wlen = chunk.size(-1)
            
            if wlen <= pooler.keep_k:
                
                for k in range(wlen):
                    frame_states[start+k] = 1
                    frame_group_ids[start+k] = start+k
                continue

            
            metric, sim = pooler._compute_sim(chunk)
            s_pred = pooler._token_scores(chunk)
            imp_norm = chunk.norm(dim=1)
            s_mix = pooler.alpha_mix * s_pred + (1 - pooler.alpha_mix) * imp_norm
            
            
            k_to_keep = min(pooler.keep_k, wlen)
            top_indices = s_mix[0].topk(k_to_keep).indices # [k]
            
            
            
            attn = torch.softmax(sim / pooler.tau, dim=-1)[0] # [wlen, wlen]
            
            
            for tip in top_indices:
                global_idx = start + tip.item()
                frame_states[global_idx] = 1 # Keeper
                frame_group_ids[global_idx] = global_idx 

            
            all_indices = torch.arange(wlen, device=chunk.device)
            
            
            for local_idx in range(wlen):
                if local_idx in top_indices: continue
                
                
                
                relevant_attn = attn[local_idx][top_indices]
                best_match_in_top = relevant_attn.argmax()
                target_local_idx = top_indices[best_match_in_top].item()
                
                global_curr = start + local_idx
                global_target = start + target_local_idx
                
                frame_states[global_curr] = 0 # Merged
                frame_group_ids[global_curr] = global_target 
                print(top_indices)

        
        axes[exp_idx, 0].set_ylabel(f"Expert {exp_idx}\n(Rate={rate})", fontsize=12, fontweight='bold')
        
        for t in range(T):
            ax = axes[exp_idx, t]
            img = raw_images[t].copy()
            
            group_id = frame_group_ids[t]
            is_rep = frame_states[t] == 1
            
            
            
            
            color_idx = (group_id * 3 + 7) % 20 
            color_rgba = cmap(color_idx) # (r, g, b, a) 0-1
            color_rgb_u8 = (int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))
            
            
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            
            
            
            
            
            overlay = np.full_like(img, color_rgb_u8, dtype=np.uint8)
            alpha = 0.3 
            blended = cv2.addWeighted(img, 1.0, overlay, alpha, 0)
            ax.imshow(blended)

            
            rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], 
                                     linewidth=0, edgecolor='none', facecolor='none')
            
            if is_rep:
                
                rect.set_linewidth(8)
                rect.set_edgecolor(color_rgba)
                # ax.text(10, 30, "REP", color='white', fontsize=10, fontweight='bold', 
                #         bbox=dict(facecolor=color_rgba, edgecolor='none', alpha=0.8))
            else:
                
                rect.set_linewidth(3)
                rect.set_edgecolor(color_rgba)
                rect.set_linestyle('--') 
            
            ax.add_patch(rect)
            
            if exp_idx == 0:
                ax.set_title(str(t), fontsize=8)

    plt.tight_layout()
    plt.savefig(save_name, dpi=100)
    plt.close()
    print(f"[INFO] Temporal merging visualization saved to {save_name}")


def visualize_merged_frames_pixel_space(model, video_head, image_input, config, device, save_name="vis_pixel_merging_result.png"):
    """
    Visualize merged frames in pixel space using weighted blending instead of additive overlay.
    """
    
    n_seg = config.data.num_segments
    if image_input.dim() == 5: image_input = image_input[0]
    target_frames = image_input[:n_seg].clone().detach().to(device)
    
    
    
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
    
    # x * std + mean -> [0, 1] -> [0, 255]
    denorm_frames = target_frames * std + mean
    denorm_frames = torch.clamp(denorm_frames, 0, 1)
    
    
    raw_images = [f.permute(1, 2, 0).cpu().numpy() * 255.0 for f in denorm_frames]
    raw_images = [img.astype(np.float32) for img in raw_images] 
    T = len(raw_images)

    
    encoder = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        features_flat, _ = encoder(target_frames)
        if features_flat.dim() == 2: features_seq = features_flat.unsqueeze(1) 
        else: features_seq = features_flat.permute(1, 0, 2)
    
    head_mod = video_head.module if hasattr(video_head, 'module') else video_head
    rates = head_mod.sampling_rates
    num_experts = len(rates)

    
    fig = plt.figure(figsize=(24, 4 * num_experts + 4))
    plt.subplots_adjust(hspace=0.6, wspace=0.1)

    print(f"[VIS] Synthesizing Merged Frames (Linear Blending Mode)...")

    
    show_indices = range(T)
    if T > 16: show_indices = np.linspace(0, T-1, 16, dtype=int)
    
    for i, idx in enumerate(show_indices):
        ax = plt.subplot2grid((num_experts + 1, len(show_indices)), (0, i))
        ax.imshow(raw_images[idx].astype(np.uint8))
        ax.axis('off')
        if i == 0: ax.set_title("Original Input", fontsize=14, fontweight='bold', loc='left')
        ax.text(0, -10, f"T{idx}", fontsize=10)

    
    for exp_idx in range(num_experts):
        pooler = head_mod.pooling_layers[exp_idx]
        rate = rates[exp_idx]
        chunk_feat = features_seq.permute(1, 2, 0) 
        
        synthesized_frames = []
        frame_timestamps = []

        
        for start in range(0, T, rate):
            end = min(start + rate, T)
            chunk = chunk_feat[:, :, start:end]
            wlen = chunk.size(-1)
            
            
            metric, sim = pooler._compute_sim(chunk)
            s_pred = pooler._token_scores(chunk)
            imp_norm = chunk.norm(dim=1)
            s_mix = pooler.alpha_mix * s_pred + (1 - pooler.alpha_mix) * imp_norm
            
            
            k_to_keep = min(pooler.keep_k, wlen)
            top_indices = s_mix[0].topk(k_to_keep).indices
            top_indices, _ = top_indices.sort()
            
            
            attn = torch.softmax(sim / pooler.tau, dim=-1)[0] 
            
            full_indices = torch.arange(wlen, device=chunk.device)
            mask = torch.ones(wlen, dtype=torch.bool, device=chunk.device)
            mask[top_indices] = False
            rest_indices = full_indices[mask]

            relevant_attn = attn[rest_indices][:, top_indices]
            
            norm_w = relevant_attn / (relevant_attn.sum(dim=1, keepdim=True) + 1e-6)
            norm_w_cpu = norm_w.detach().cpu().numpy()
            
            
            for k_local_idx in range(len(top_indices)):
                curr_keep_idx = top_indices[k_local_idx].item()
                base_img = raw_images[start + curr_keep_idx].copy() 
                
                ghost_img = np.zeros_like(base_img)
                
                
                if len(rest_indices) > 0:
                    for r_local_idx in range(len(rest_indices)):
                        curr_rest_idx = rest_indices[r_local_idx].item()
                        weight = norm_w_cpu[r_local_idx, k_local_idx]
                        ghost_img += weight * raw_images[start + curr_rest_idx]
                    
                    
                    
                    
                    final_img = cv2.addWeighted(base_img, 0.6, ghost_img, 0.4, 0)
                else:
                    
                    final_img = base_img
                
                synthesized_frames.append(final_img.astype(np.uint8))
                frame_timestamps.append(start + curr_keep_idx)

        
        n_out = len(synthesized_frames)
        display_indices = range(n_out)
        if n_out > 16: 
            display_indices = np.linspace(0, n_out-1, 16, dtype=int)
        
        for i, idx in enumerate(display_indices):
            ax = plt.subplot2grid((num_experts + 1, len(show_indices)), (exp_idx + 1, i))
            
            ax.imshow(synthesized_frames[idx])
            ax.axis('off')
            
            if i == 0:
                label = f"Expert {exp_idx}\nRate {rate}\n({n_out} frames)"
                ax.text(-10, 50, label, fontsize=12, fontweight='bold', va='center', rotation=90)
            
            ax.text(0, -10, f"Base:T{frame_timestamps[idx]}", fontsize=9, color='darkblue', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_name, dpi=100)
    plt.close()
    print(f"[INFO] Pixel merging visualization saved to {save_name}")


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def visualize_all_frames_patch_grid1(model, video_head, image_input, config, device, save_name="vis_all_frames_grid.png"):
    """
    Visualize patch-grid saliency for all frames in the full sequence.
    The visualization keeps the full timeline, uses percentile normalization, and applies gamma enhancement.
    """
    
    n_seg = config.data.num_segments
    if image_input.dim() == 5: image_input = image_input[0]
    
    
    
    T_total = image_input.shape[0]
    
    target_frames = image_input.clone().detach().to(device)
    target_frames.requires_grad = True
    
    
    denorm_frames = unnormalize_tensor_robust(target_frames) 
    raw_images = [f.detach().permute(1, 2, 0).cpu().numpy() * 255.0 for f in denorm_frames]
    raw_images = [img.astype(np.uint8) for img in raw_images]
    T = len(raw_images)
    H, W = target_frames.shape[-2:]
    
    patch_size = 16
    
    encoder = model.module if hasattr(model, 'module') else model
    head_mod = video_head.module if hasattr(video_head, 'module') else video_head
    rates = head_mod.sampling_rates
    num_experts = len(rates)
    
    results_grid = {} 

    print(f"[VIS] Computing Full Sequence Patch Saliency ({T} frames)...")

    
    for exp_idx in range(num_experts):
        if target_frames.grad is not None: target_frames.grad.zero_()
        
        # Forward
        features_flat, _ = encoder(target_frames)
        if features_flat.dim() == 2: features_seq = features_flat.unsqueeze(1) 
        else: features_seq = features_flat.permute(1, 0, 2)
        
        pooler = head_mod.pooling_layers[exp_idx]
        feat_in = features_seq.permute(1, 2, 0) 
        pooled_out, _ = pooler(feat_in)
        
        loss = pooled_out.norm()
        loss.backward()
        
        grads = target_frames.grad.data 
        pixel_saliency = grads.abs().max(dim=1)[0] # [T, H, W]
        
        # Patch Aggregation
        patch_saliency = F.avg_pool2d(
            pixel_saliency.unsqueeze(1), 
            kernel_size=patch_size, 
            stride=patch_size
        ).squeeze(1).cpu().numpy() # [T, 14, 14]
        
        results_grid[exp_idx] = patch_saliency

    
    
    fig_w = max(20, T * 0.8)
    fig_h = num_experts * 1.1
    
    
    fig, axes = plt.subplots(num_experts, T, figsize=(fig_w, fig_h), dpi=150)
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    if num_experts == 1: axes = axes[np.newaxis, :]

    for row, exp_idx in enumerate(range(num_experts)):
        rate = rates[exp_idx]
        grid_maps = results_grid[exp_idx] # [T, 14, 14]
        
        
        
        robust_max = np.percentile(grid_maps, 99) + 1e-8
        
        axes[row, 0].set_ylabel(f"Expert {exp_idx}", fontsize=12, fontweight='bold')

        for col in range(T):
            ax = axes[row, col]
            
            img = raw_images[col]
            patch_map = grid_maps[col]
            
            # Resize
            patch_map_resized = cv2.resize(patch_map, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # Normalize & Boost
            norm_map = patch_map_resized / robust_max
            norm_map = np.clip(norm_map, 0, 1)
            
            norm_map = np.power(norm_map, 0.6) 
            
            # Heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * norm_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
            
            # Grid Lines (Same subtle logic)
            grid_layer = np.zeros_like(overlay)
            grid_color = (220, 220, 220)
            for y in range(0, H, patch_size):
                cv2.line(grid_layer, (0, y), (W, y), grid_color, 1)
            for x in range(0, W, patch_size):
                cv2.line(grid_layer, (x, 0), (x, H), grid_color, 1)
            
            mask = np.any(grid_layer > 0, axis=-1)
            overlay[mask] = cv2.addWeighted(overlay[mask], 0.6, grid_layer[mask], 0.4, 0)

            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            
            
            if row == 0 and (col % 4 == 0):
                ax.set_title(f"T{col}", fontsize=10)

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[INFO] All-Frames Grid visualization saved to {save_path}")

def visualize_all_frames_patch_grid2(model, video_head, image_input, config, device,
                                     save_name="vis_all_frames_grid2.png"):
    """
    Visualize patch-grid saliency for selected frames of each expert with a compact layout.
    """
    
    n_seg = config.data.num_segments
    if image_input.dim() == 5:
        image_input = image_input[0]  # [T, C, H, W]

    target_frames = image_input.clone().detach().to(device)
    target_frames.requires_grad = True

    denorm_frames = unnormalize_tensor_robust(target_frames)
    raw_images = [
        (f.detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        for f in denorm_frames
    ]

    T = len(raw_images)
    H, W = target_frames.shape[-2:]
    patch_size = 16

    encoder = model.module if hasattr(model, 'module') else model
    head_mod = video_head.module if hasattr(video_head, 'module') else video_head
    rates = head_mod.sampling_rates
    num_experts = len(rates)

    
    experts = {
        '0': [0, 3, 4, 7, 8, 10, 13, 14, 16, 18, 21, 23, 24, 27, 29, 30],
        '1': [0, 4, 10, 14, 18, 24, 29],
        '2': [0, 10, 21, 25],
        '3': [0, 20],
    }
    
    max_keep = max(len(v) for v in experts.values())

    
    results_grid = {}

    print(f"[VIS] Computing Full Sequence Patch Saliency ({T} frames) for {num_experts} experts...")

    for exp_idx in range(num_experts):
        if target_frames.grad is not None:
            target_frames.grad.zero_()

        features_flat, _ = encoder(target_frames)
        if features_flat.dim() == 2:
            features_seq = features_flat.unsqueeze(1)
        else:
            features_seq = features_flat.permute(1, 0, 2)

        pooler = head_mod.pooling_layers[exp_idx]
        feat_in = features_seq.permute(1, 2, 0)   # [B, D, T]
        pooled_out, _ = pooler(feat_in)

        loss = pooled_out.norm()
        loss.backward()

        grads = target_frames.grad.data
        pixel_saliency = grads.abs().max(dim=1)[0]

        patch_saliency = F.avg_pool2d(
            pixel_saliency.unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size
        ).squeeze(1).cpu().numpy()   # [T, H_patch, W_patch]

        results_grid[exp_idx] = patch_saliency

    
    fig_w = max(12, max_keep * 0.8)
    fig_h = num_experts * 1.0

    fig, axes = plt.subplots(num_experts, max_keep,
                             figsize=(fig_w, fig_h), dpi=400)
    
    plt.subplots_adjust(left=0.06, right=1, top=1, bottom=0,
                        wspace=0.02, hspace=0.02)

    if num_experts == 1:
        axes = axes[np.newaxis, :]

    
    for ax in axes.flat:
        ax.axis('off')

    for row, exp_idx in enumerate(range(num_experts)):
        grid_maps = results_grid[exp_idx]
        robust_max = np.percentile(grid_maps, 99) + 1e-8

        T_ex = experts[str(exp_idx)]
        if len(T_ex) == 0:
            continue

        
        axes[row, 0].set_ylabel(f"Expert {exp_idx}", fontsize=11)

        for col_idx, t in enumerate(T_ex):
            if col_idx >= max_keep:
                break
            if t >= T:
                continue  

            ax = axes[row, col_idx]
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])

            img = raw_images[t]
            patch_map = grid_maps[t]

            
            patch_map_resized = cv2.resize(
                patch_map.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_CUBIC
            )

            
            ph, pw = patch_map.shape
            y, x = np.mgrid[0:ph, 0:pw]
            cy = cx = ph / 2.0
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            base_sigma = ph / 4.0
            if num_experts > 1:
                
                scale_min, scale_max = 0.7, 1.5
                scale = scale_min + (scale_max - scale_min) * (exp_idx / (num_experts - 1))
            else:
                scale = 1.0
            sigma = base_sigma * scale

            gaussian_mask_patch = np.exp(- dist ** 2 / (2 * sigma * sigma))
            gaussian_mask_patch /= gaussian_mask_patch.max()

            gaussian_mask = cv2.resize(
                gaussian_mask_patch.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_CUBIC
            )

            patch_map_resized = patch_map_resized * gaussian_mask

            
            sigma_gauss = 10.0
            patch_map_resized = cv2.GaussianBlur(
                patch_map_resized,
                ksize=(0, 0),
                sigmaX=sigma_gauss,
                sigmaY=sigma_gauss
            )

            # === Normalize + Gamma ===
            norm_map = np.clip(patch_map_resized / robust_max, 0, 1)
            norm_map = np.power(norm_map, 0.6)

            # Heatmap
            heatmap = cv2.applyColorMap(
                (norm_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Overlay
            overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

            ax.imshow(overlay)

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"[INFO] All-Frames Grid2 visualization saved to {save_path}")


import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize_all_frames_patch_grid(model, video_head, image_input, config, device,
                                    save_name="vis_all_frames_grid.png"):
    """
    Visualize patch-grid saliency only for representative frames kept by each expert.
    """
    
    n_seg = config.data.num_segments
    if image_input.dim() == 5:
        image_input = image_input[0]  # [T, C, H, W]

    target_frames = image_input.clone().detach().to(device)
    target_frames.requires_grad = True

    denorm_frames = unnormalize_tensor_robust(target_frames)
    raw_images = [
        (f.detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        for f in denorm_frames
    ]

    T = len(raw_images)
    H, W = target_frames.shape[-2:]
    patch_size = 16

    encoder = model.module if hasattr(model, 'module') else model
    head_mod = video_head.module if hasattr(video_head, 'module') else video_head
    rates = head_mod.sampling_rates
    num_experts = len(rates)

    
    with torch.no_grad():
        features_flat, _ = encoder(target_frames)  
        if features_flat.dim() == 2:
            features_seq = features_flat.unsqueeze(1)      # [T, 1, D]
        else:
            features_seq = features_flat.permute(1, 0, 2)  # [T, B, D]

        
        chunk_feat_full = features_seq.permute(1, 2, 0)    # [B, D, T]

    keep_indices = {}   # {exp_idx: [kept_t1, kept_t2, ...]}
    max_keep = 0

    for exp_idx, rate in enumerate(rates):
        pooler = head_mod.pooling_layers[exp_idx]
        keep_mask = np.zeros(T, dtype=bool)

        for start in range(0, T, rate):
            end = min(start + rate, T)
            chunk = chunk_feat_full[:, :, start:end]   # [B, D, wlen]
            wlen = chunk.size(-1)

            if wlen <= pooler.keep_k:
                
                keep_mask[start:end] = True
                continue

            
            metric, sim = pooler._compute_sim(chunk)          
            s_pred = pooler._token_scores(chunk)              # [B, wlen]
            imp_norm = chunk.norm(dim=1)                      # [B, wlen]
            s_mix = pooler.alpha_mix * s_pred + (1 - pooler.alpha_mix) * imp_norm

            k_to_keep = min(pooler.keep_k, wlen)
            top_indices = s_mix[0].topk(k_to_keep).indices    # [k]

            for tip in top_indices:
                keep_mask[start + tip.item()] = True

        kept = np.where(keep_mask)[0].tolist()
        keep_indices[exp_idx] = kept
        max_keep = max(max_keep, len(kept))

    if max_keep == 0:
        print("[WARN] No kept frames found, nothing to visualize.")
        return

    
    results_grid = {}

    print(f"[VIS] Computing Patch Saliency for {T} frames, "
          f"but only plotting kept frames (max {max_keep} per expert)...")

    for exp_idx in range(num_experts):
        if target_frames.grad is not None:
            target_frames.grad.zero_()

        features_flat, _ = encoder(target_frames)
        if features_flat.dim() == 2:
            features_seq = features_flat.unsqueeze(1)
        else:
            features_seq = features_flat.permute(1, 0, 2)

        pooler = head_mod.pooling_layers[exp_idx]
        feat_in = features_seq.permute(1, 2, 0)   # [B, D, T]
        pooled_out, _ = pooler(feat_in)

        loss = pooled_out.norm()
        loss.backward()

        grads = target_frames.grad.data
        pixel_saliency = grads.abs().max(dim=1)[0]

        patch_saliency = F.avg_pool2d(
            pixel_saliency.unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size
        ).squeeze(1).cpu().numpy()   # [T, H_patch, W_patch]

        results_grid[exp_idx] = patch_saliency

    
    fig_w = max(12, max_keep * 0.8)
    fig_h = num_experts * 1.01

    fig, axes = plt.subplots(num_experts, max_keep,
                             figsize=(fig_w, fig_h), dpi=400)
    # plt.subplots_adjust(wspace=0.01, hspace=0.15)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0.02, hspace=0.02)
    
    if num_experts == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        ax.axis('off')

    for row, exp_idx in enumerate(range(num_experts)):
        grid_maps = results_grid[exp_idx]
        robust_max = np.percentile(grid_maps, 99) + 1e-8

        kept_ts = keep_indices[exp_idx]   
        axes[row, 0].set_ylabel(f"Expert {exp_idx}", fontsize=11)

        for col_idx, t in enumerate(kept_ts):
            if col_idx >= max_keep:
                break
            ax = axes[row, col_idx]

            img = raw_images[t]
            patch_map = grid_maps[t]

            
            patch_map_resized = cv2.resize(
                patch_map,
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )

            
            ph, pw = patch_map.shape
            y, x = np.mgrid[0:ph, 0:pw]
            cy = cx = ph / 2.0
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            
            base_sigma = ph / 4

            if num_experts > 1:
                
                
                scale_min, scale_max = 0.7, 1.5
                scale = scale_min + (scale_max - scale_min) * (exp_idx / (num_experts - 1))
            else:
                scale = 1.0

            sigma = base_sigma * scale  
            # print(f"[DEBUG] Expert {exp_idx} using sigma={sigma:.2f} for Gaussian mask.")
            gaussian_mask_patch = np.exp(- dist ** 2 / (2 * sigma * sigma))
            gaussian_mask_patch /= gaussian_mask_patch.max()

            gaussian_mask = cv2.resize(
                gaussian_mask_patch,
                (W, H),
                interpolation=cv2.INTER_CUBIC
            )

            patch_map_resized = patch_map_resized * gaussian_mask

            sigma_gauss = 10.0
            patch_map_resized = cv2.GaussianBlur(
                patch_map_resized,
                ksize=(0, 0),
                sigmaX=sigma_gauss,
                sigmaY=sigma_gauss
            )

            # === Normalize + Gamma ===
            norm_map = np.clip(patch_map_resized / robust_max, 0, 1)
            norm_map = np.power(norm_map, 0.6)

            # Heatmap
            heatmap = cv2.applyColorMap(
                (norm_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Overlay
            overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

            # Grid lines
            # grid_layer = np.zeros_like(overlay)
            # grid_color = (220, 220, 220)
            # for yy in range(0, H, patch_size):
            #     cv2.line(grid_layer, (0, yy), (W, yy), grid_color, 1)
            # for xx in range(0, W, patch_size):
            #     cv2.line(grid_layer, (xx, 0), (xx, H), grid_color, 1)

            # mask_grid = np.any(grid_layer > 0, axis=-1)
            # overlay[mask_grid] = cv2.addWeighted(
            #     overlay[mask_grid], 0.6,
            #     grid_layer[mask_grid], 0.4,
            #     0
            # )

            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('on')

            
            # if row == 0:
            #     ax.set_title(f"T{t}", fontsize=9)

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"[INFO] All-Frames Grid visualization saved to {save_path}")

def visualize_all_frames_patch_grid_original(model, video_head, image_input, config, device,
                                    save_name="vis_all_frames_grid.png"):
    """
    Visualize representative-frame patch grids with the original multi-row layout.
    The expert with the most kept frames gets one raw-image row plus one heatmap row; other experts use a single heatmap row.
    """
    
    n_seg = config.data.num_segments
    if image_input.dim() == 5:
        image_input = image_input[0]  # [T, C, H, W]

    target_frames = image_input.clone().detach().to(device)
    target_frames.requires_grad = True

    denorm_frames = unnormalize_tensor_robust(target_frames)
    raw_images = [
        (f.detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        for f in denorm_frames
    ]

    T = len(raw_images)
    H, W = target_frames.shape[-2:]
    patch_size = 16

    encoder = model.module if hasattr(model, 'module') else model
    head_mod = video_head.module if hasattr(video_head, 'module') else video_head
    rates = head_mod.sampling_rates
    num_experts = len(rates)

    
    with torch.no_grad():
        features_flat, _ = encoder(target_frames)  
        if features_flat.dim() == 2:
            features_seq = features_flat.unsqueeze(1)      # [T, 1, D]
        else:
            features_seq = features_flat.permute(1, 0, 2)  # [T, B, D]

        
        chunk_feat_full = features_seq.permute(1, 2, 0)    # [B, D, T]

    keep_indices = {}   # {exp_idx: [kept_t1, kept_t2, ...]}
    max_keep = 0

    for exp_idx, rate in enumerate(rates):
        pooler = head_mod.pooling_layers[exp_idx]
        keep_mask = np.zeros(T, dtype=bool)

        for start in range(0, T, rate):
            end = min(start + rate, T)
            chunk = chunk_feat_full[:, :, start:end]   # [B, D, wlen]
            wlen = chunk.size(-1)

            if wlen <= pooler.keep_k:
                keep_mask[start:end] = True
                continue

            metric, sim = pooler._compute_sim(chunk)
            s_pred = pooler._token_scores(chunk)              # [B, wlen]
            imp_norm = chunk.norm(dim=1)                      # [B, wlen]
            s_mix = pooler.alpha_mix * s_pred + (1 - pooler.alpha_mix) * imp_norm

            k_to_keep = min(pooler.keep_k, wlen)
            top_indices = s_mix[0].topk(k_to_keep).indices    # [k]

            for tip in top_indices:
                keep_mask[start + tip.item()] = True

        kept = np.where(keep_mask)[0].tolist()
        keep_indices[exp_idx] = kept
        max_keep = max(max_keep, len(kept))

    if max_keep == 0:
        print("[WARN] No kept frames found, nothing to visualize.")
        return

    
    sorted_experts = sorted(range(num_experts),
                            key=lambda e: len(keep_indices[e]),
                            reverse=True)
    main_expert = sorted_experts[0]          
    main_keep_num = len(keep_indices[main_expert])

    
    results_grid = {}

    print(f"[VIS] Computing Patch Saliency for {T} frames, "
          f"max_keep = {max_keep} ...")

    for exp_idx in range(num_experts):
        if target_frames.grad is not None:
            target_frames.grad.zero_()

        features_flat, _ = encoder(target_frames)
        if features_flat.dim() == 2:
            features_seq = features_flat.unsqueeze(1)
        else:
            features_seq = features_flat.permute(1, 0, 2)

        pooler = head_mod.pooling_layers[exp_idx]
        feat_in = features_seq.permute(1, 2, 0)   # [B, D, T]
        pooled_out, _ = pooler(feat_in)

        loss = pooled_out.norm()
        loss.backward()

        grads = target_frames.grad.data
        pixel_saliency = grads.abs().max(dim=1)[0]

        patch_saliency = F.avg_pool2d(
            pixel_saliency.unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size
        ).squeeze(1).cpu().numpy()   # [T, H_patch, W_patch]

        results_grid[exp_idx] = patch_saliency

    
    
    rows_total = 1 + num_experts
    fig_w = max(12, max_keep * 0.8)
    fig_h = rows_total * 0.9

    fig, axes = plt.subplots(rows_total, max_keep,
                             figsize=(fig_w, fig_h), dpi=400)
    plt.subplots_adjust(left=0.08, right=1, top=1, bottom=0,
                        wspace=0.02, hspace=0.05)

    
    if rows_total == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        ax.axis('off')

    
    kept_ts_main = keep_indices[main_expert]
    grid_maps_main = results_grid[main_expert]
    robust_max_main = np.percentile(grid_maps_main, 99) + 1e-8

    
    axes[1, 0].set_ylabel(f"{main_keep_num}-Frames Expert", fontsize=11)

    for col_idx, t in enumerate(kept_ts_main):
        if col_idx >= max_keep or t >= T:
            continue

        
        ax_orig = axes[0, col_idx]
        ax_orig.axis('on')
        ax_orig.imshow(raw_images[t])
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        
        ax = axes[1, col_idx]
        ax.axis('on')

        img = raw_images[t]
        patch_map = grid_maps_main[t]

        
        patch_map_resized = cv2.resize(
            patch_map,
            (W, H),
            interpolation=cv2.INTER_NEAREST
        )

        
        ph, pw = patch_map.shape
        y, x = np.mgrid[0:ph, 0:pw]
        cy = cx = ph / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        base_sigma = ph / 4
        if num_experts > 1:
            scale_min, scale_max = 0.7, 1.5
            scale = scale_min + (scale_max - scale_min) * (main_expert / (num_experts - 1))
        else:
            scale = 1.0

        sigma = base_sigma * scale
        gaussian_mask_patch = np.exp(- dist ** 2 / (2 * sigma * sigma))
        gaussian_mask_patch /= gaussian_mask_patch.max()

        gaussian_mask = cv2.resize(
            gaussian_mask_patch,
            (W, H),
            interpolation=cv2.INTER_CUBIC
        )

        patch_map_resized = patch_map_resized * gaussian_mask

        sigma_gauss = 10.0
        patch_map_resized = cv2.GaussianBlur(
            patch_map_resized,
            ksize=(0, 0),
            sigmaX=sigma_gauss,
            sigmaY=sigma_gauss
        )

        # === Normalize + Gamma ===
        norm_map = np.clip(patch_map_resized / robust_max_main, 0, 1)
        norm_map = np.power(norm_map, 0.6)

        # Heatmap
        heatmap = cv2.applyColorMap(
            (norm_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
        ax.imshow(overlay)
        ax.set_xticks([])
        ax.set_yticks([])

    
    current_row = 2  
    for exp_idx in sorted_experts[1:]:
        kept_ts = keep_indices[exp_idx]
        n_kept = len(kept_ts)
        if n_kept == 0:
            continue

        grid_maps = results_grid[exp_idx]
        robust_max = np.percentile(grid_maps, 99) + 1e-8

        
        axes[current_row, 0].set_ylabel(f"{n_kept}-Frames Expert", fontsize=11)

        for col_idx, t in enumerate(kept_ts):
            if col_idx >= max_keep or t >= T:
                continue

            ax = axes[current_row, col_idx]
            ax.axis('on')

            img = raw_images[t]
            patch_map = grid_maps[t]

            
            patch_map_resized = cv2.resize(
                patch_map,
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )

            
            ph, pw = patch_map.shape
            y, x = np.mgrid[0:ph, 0:pw]
            cy = cx = ph / 2.0
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            base_sigma = ph / 4
            if num_experts > 1:
                scale_min, scale_max = 0.7, 1.5
                scale = scale_min + (scale_max - scale_min) * (exp_idx / (num_experts - 1))
            else:
                scale = 1.0

            sigma = base_sigma * scale
            gaussian_mask_patch = np.exp(- dist ** 2 / (2 * sigma * sigma))
            gaussian_mask_patch /= gaussian_mask_patch.max()

            gaussian_mask = cv2.resize(
                gaussian_mask_patch,
                (W, H),
                interpolation=cv2.INTER_CUBIC
            )

            patch_map_resized = patch_map_resized * gaussian_mask

            sigma_gauss = 10.0
            patch_map_resized = cv2.GaussianBlur(
                patch_map_resized,
                ksize=(0, 0),
                sigmaX=sigma_gauss,
                sigmaY=sigma_gauss
            )

            # === Normalize + Gamma ===
            norm_map = np.clip(patch_map_resized / robust_max, 0, 1)
            norm_map = np.power(norm_map, 0.6)

            heatmap = cv2.applyColorMap(
                (norm_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])

        current_row += 1

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"[INFO] All-Frames Grid visualization saved to {save_path}")


def visualize_kept_frames_compact(model, video_head, image_input, config, device, save_name="vis_kept_frames_compact.png"):
    """
    Compact visualization of representative-frame patch saliency.
    This version keeps the encoder forward pass inside the computation graph.
    """
    
    n_seg = config.data.num_segments
    if image_input.dim() == 5: image_input = image_input[0]
    
    
    target_frames = image_input[:n_seg].clone().detach().to(device)
    target_frames.requires_grad = True
    
    
    denorm_frames = unnormalize_tensor_robust(target_frames) 
    raw_images = [f.detach().permute(1, 2, 0).cpu().numpy() * 255.0 for f in denorm_frames]
    raw_images = [img.astype(np.uint8) for img in raw_images]
    T = len(raw_images)
    H, W = target_frames.shape[-2:]
    
    patch_size = 16
    
    encoder = model.module if hasattr(model, 'module') else model
    head_mod = video_head.module if hasattr(video_head, 'module') else video_head
    rates = head_mod.sampling_rates
    num_experts = len(rates)
    
    
    min_rate = min(rates)
    max_cols = (T + min_rate - 1) // min_rate

    results_grid = {} 

    print(f"[VIS] Computing Saliency for Kept Frames Only...")

    
    for exp_idx in range(num_experts):
        
        if target_frames.grad is not None: target_frames.grad.zero_()
        
        
        
        features_flat, _ = encoder(target_frames)
        if features_flat.dim() == 2: features_seq = features_flat.unsqueeze(1) 
        else: features_seq = features_flat.permute(1, 0, 2)
        
        pooler = head_mod.pooling_layers[exp_idx]
        feat_in = features_seq.permute(1, 2, 0) 
        
        # Pooling Forward
        pooled_out, _ = pooler(feat_in)
        
        # Backward
        loss = pooled_out.norm()
        loss.backward() 
        
        
        grads = target_frames.grad.data 
        pixel_saliency = grads.abs().max(dim=1)[0]
        
        # Patch Aggregation
        patch_saliency = F.avg_pool2d(
            pixel_saliency.unsqueeze(1), 
            kernel_size=patch_size, 
            stride=patch_size
        ).squeeze(1).cpu().numpy() # [T, 14, 14]
        
        results_grid[exp_idx] = patch_saliency

    
    fig, axes = plt.subplots(num_experts, max_cols, figsize=(max_cols * 1.2, num_experts * 1.6), dpi=150)
    plt.subplots_adjust(wspace=0.02, hspace=0.05) 
    if num_experts == 1: axes = axes[np.newaxis, :]

    
    
    
    with torch.no_grad():
        
        features_flat, _ = encoder(target_frames)
        if features_flat.dim() == 2: features_seq = features_flat.unsqueeze(1) 
        else: features_seq = features_flat.permute(1, 0, 2)

        for row, exp_idx in enumerate(range(num_experts)):
            rate = rates[exp_idx]
            pooler = head_mod.pooling_layers[exp_idx]
            chunk_feat = features_seq.permute(1, 2, 0) # [1, D, T]
            grid_maps = results_grid[exp_idx]
            
            
            kept_indices = []
            for start in range(0, T, rate):
                end = min(start + rate, T)
                chunk = chunk_feat[:, :, start:end]
                
                
                metric, sim = pooler._compute_sim(chunk)
                s_pred = pooler._token_scores(chunk)
                imp_norm = chunk.norm(dim=1)
                s_mix = pooler.alpha_mix * s_pred + (1 - pooler.alpha_mix) * imp_norm
                
                k_to_keep = min(pooler.keep_k, chunk.size(-1))
                top_idx = s_mix[0].topk(k_to_keep).indices.item()
                
                global_idx = start + top_idx
                kept_indices.append(global_idx)
            
            
            axes[row, 0].set_ylabel(f"Expert {exp_idx}", fontsize=12, fontweight='bold')
            
            
            relevant_maps = grid_maps[kept_indices]
            robust_max = np.percentile(relevant_maps, 99) + 1e-8

            for col in range(max_cols):
                ax = axes[row, col]
                
                
                if col >= len(kept_indices):
                    ax.axis('off')
                    continue
                
                t_idx = kept_indices[col]
                img = raw_images[t_idx]
                patch_map = grid_maps[t_idx]
                
                # Resize & Norm
                patch_map_resized = cv2.resize(patch_map, (W, H), interpolation=cv2.INTER_NEAREST)
                norm_map = patch_map_resized / robust_max
                norm_map = np.clip(norm_map, 0, 1)
                norm_map = np.power(norm_map, 0.6)
                
                heatmap = cv2.applyColorMap(np.uint8(255 * norm_map), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
                
                # Grid Lines
                grid_layer = np.zeros_like(overlay)
                grid_color = (220, 220, 220)
                for y in range(0, H, patch_size): cv2.line(grid_layer, (0, y), (W, y), grid_color, 1)
                for x in range(0, W, patch_size): cv2.line(grid_layer, (x, 0), (x, H), grid_color, 1)
                mask = np.any(grid_layer > 0, axis=-1)
                
                blended_pixels = cv2.addWeighted(overlay[mask], 0.7, grid_layer[mask], 0.3, 0)
                overlay[mask] = blended_pixels

                ax.imshow(overlay)
                ax.set_xticks([])
                ax.set_yticks([])
                
                
                # ax.text(5, 20, f"T{t_idx}", color='white', fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.close()
    print(f"[INFO] Compact Kept-Frames Visualization saved to {save_name}")


def unnormalize_tensor_robust(tensor):
    res = []
    for t in range(tensor.shape[0]):
        img = tensor[t]
        g_min, g_max = img.min(), img.max()
        if g_max > g_min:
            img = (img - g_min) / (g_max - g_min)
        res.append(img)
    return torch.stack(res)



def unnormalize_tensor_robust(tensor):
    # tensor: [T, C, H, W]
    
    res = []
    for t in range(tensor.shape[0]):
        img = tensor[t]
        g_min, g_max = img.min(), img.max()
        if g_max > g_min:
            img = (img - g_min) / (g_max - g_min)
        res.append(img)
    return torch.stack(res)
# ==============================================================
# 2. Full validate function
# ==============================================================

def validate(val_loader, device, model, video_head, config, text_features,
             test_crops, test_clips, distributed=False, rank=0):
    """
    Distributed validation with multi-crop/clip averaging + Half-Class Evaluation + Spatial Visualization
    """
    model.eval()
    video_head.eval()
    proc_start_time = time.time()
    autocast_ctx = torch.cuda.amp.autocast
    all_preds, all_labels = [], []
    total_correct1, total_correct5, total_count = 0.0, 0.0, 0

    # Collect gating_weights, class_id, and vis_embs
    all_gating_weights = []
    all_class_ids = []
    all_vis_embs = []  

    
    with torch.no_grad():
        n_class = text_features.size(0)

        for i, (image, class_id) in enumerate(val_loader):
            
            # ============================================================
            
            # ============================================================
            
            if rank == 0:
                try:
                    print("[INFO] Running Full Spatial Visualization on first batch...")
                    with torch.enable_grad():
                        
                        n_frames_vis = config.data.num_segments 
                        
                        # image shape: [B, num_crops*T, C, H, W]
                        
                        
                        
                        
                        
                        
                        vis_input_raw = image.view(-1, 3, image.size(-2), image.size(-1)) # [B*T_total, C, H, W]
                        
                        
                        vis_input = vis_input_raw[:n_frames_vis] # [32, C, H, W]
                        vis_input = vis_input.to(device)

                        vis_root = '/home/linrui/moe/TimeMoE/vis_supplementary_train'
                        save_patch_sal2 = os.path.join(
                            vis_root, f"vis_patch_saliency2_batch{i:04d}.png"
                        )
                        
                        # run_full_spatial_visualization(
                        #     model=model, 
                        #     video_head=video_head, 
                        #     image_input=vis_input, 
                        #     config=config, 
                        #     device=device,
                        #     save_name="vis_all_experts_32frames.png"
                        # )
                        
                        # visualize_temporal_token_merging(
                        #     model=model,
                        #     video_head=video_head,
                        
                        #     config=config,
                        #     device=device,
                        #     save_name="vis_tome_merging.png"
                        # )
                        
                        #     model=model,
                        #     video_head=video_head,
                        #     image_input=vis_input,
                        #     config=config,
                        #     device=device,
                        #     save_name="vis_pixel_merging.png"
                        # )
                        
                        #     model=model,
                        #     video_head=video_head,
                        #     image_input=vis_input,
                        #     config=config,
                        #     device=device,
                        #     save_name="vis_patch_saliency1.png"
                        # )

                        visualize_all_frames_patch_grid_original( 
                            model=model,
                            video_head=video_head,
                            image_input=vis_input,
                            config=config,
                            device=device,
                            save_name=save_patch_sal2
                        )
                except Exception as e:
                    print(f"[WARNING] Visualization failed: {e}")
                    traceback.print_exc()
            # ============================================================
            
            # ============================================================
            batch_size = class_id.numel()
            num_crop = test_crops * test_clips
            class_id = class_id.to(device, non_blocking=True)
            n_seg = config.data.num_segments

            # reshape image batch: [batch × num_crop, num_segments, 3, H, W]
            image = image.view((-1, n_seg, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            actual_num_crop = b // batch_size
            image_input = image.to(device, non_blocking=True).view(-1, c, h, w)

            # forward
            with autocast_ctx():
                if distributed:
                    image_features, _ = model.module(image_input)
                else:
                    image_features, _ = model(image_input)
                
                
                
                
                
                
                logits, projected_features, gating_weights, div_loss, extra_loss = video_head(
                    image_features, b, t, text_embeds=text_features
                )
                
                
                all_vis_embs.append(projected_features.detach().cpu())

            
            all_gating_weights.append(gating_weights.detach().cpu())
            all_class_ids.append(class_id.detach().cpu())

            # multi-crop averaging logits
            num_classes = logits.size(-1)
            logits = logits.view(batch_size, actual_num_crop, num_classes).mean(dim=1)
            similarity = F.softmax(logits, dim=-1)

            # top-k accuracy
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
            all_preds.append(pred[0].cpu())
            all_labels.append(class_id.cpu())

            if rank == 0 and (i % config.logging.print_freq == 0):
                local_top1 = 100.0 * correct1 / bs
                local_top5 = 100.0 * correct5 / bs
                runtime = float(time.time() - proc_start_time) / (i + 1) / bs
                print(f"Test: [{i}/{len(val_loader)}], average {runtime:.4f}s/video, "
                      f"Prec@1 {local_top1:.3f}, Prec@5 {local_top5:.3f}")

    # Aggregate distributed results
    if distributed and dist.is_available() and dist.is_initialized():
        t1, t5, tc = map(lambda x: torch.tensor([x], device=device), [total_correct1, total_correct5, total_count])
        dist.all_reduce(t1, op=dist.ReduceOp.SUM)
        dist.all_reduce(t5, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
        total_correct1, total_correct5, total_count = t1.item(), t5.item(), int(tc.item())

    # Compute the final average top1/top5
    top1_avg = 100.0 * total_correct1 / max(1, total_count)
    top5_avg = 100.0 * total_correct5 / max(1, total_count)

    classes = None 

    if rank == 0:
        print(f"\n=== Full-Class Evaluation ===")
        print(f"Prec@1 {top1_avg:.3f}, Prec@5 {top5_avg:.3f}\n")

        # =========================
        # 1. Confusion matrix
        # =========================
        try:
            all_preds_tensor = torch.cat(all_preds)
            all_labels_tensor = torch.cat(all_labels)
            cm = confusion_matrix(all_labels_tensor.numpy(), all_preds_tensor.numpy())
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, ax=ax, cmap="Blues", annot=False, fmt='d')
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=400)
            plt.close()
            print("[INFO] Confusion matrix saved to confusion_matrix.png")
        except Exception as e:
            print(f"[WARN] Failed to plot confusion matrix: {e}")

        # =========================
        # 2. Expert-usage heatmap (correct vs wrong)
        # =========================
        try:
            all_gating_weights = torch.cat(all_gating_weights, dim=0)
            all_class_ids = torch.cat(all_class_ids, dim=0)
            print("[INFO] Saving class-wise expert usage heatmap...")
            
            
            # classes = visualize_expert_correct_wrong_per_class(
            #     gating_weights=all_gating_weights,
            #     preds=torch.cat(all_preds, dim=0),
            #     labels=torch.cat(all_labels, dim=0),
            #     save_dir="expert_usage_correct_wrong"
            # )
        except Exception as e:
            print(f"[WARN] Failed to plot expert usage: {e}")

        # =========================
        # 3. Half-class evaluation
        # =========================
        print("=== Half-Class Evaluation ===")
        try:
            all_vis_embs_tensor = torch.cat(all_vis_embs, dim=0).float()
            
            # acc_split, acc5_split = multi_split_test(
            #     vis_embs=all_vis_embs_tensor,
            #     text_embs=text_features.cpu().float(),
            #     true_label=all_labels_tensor
            # )
            # print("Top1 of 10 splits:", np.round(acc_split, 3))
            # print("Top5 of 10 splits:", np.round(acc5_split, 3))
            # print(f"Half-Class Mean Top1: {acc_split.mean():.3f}, Mean Top5: {acc5_split.mean():.3f}\n")
        except NameError:
            print("[INFO] 'multi_split_test' function not found, skipping half-class eval.")
        except Exception as e:
            print(f"[WARN] Half-class evaluation failed: {e}")

    return top1_avg, top5_avg, classes




# def validate(val_loader, device, model, video_head, config, text_features,
#              test_crops, test_clips, distributed=False, rank=0):
#     """Distributed validation with multi-crop/clip averaging + Half-Class Evaluation"""
#     model.eval()
#     video_head.eval()
#     proc_start_time = time.time()
#     autocast_ctx = torch.cuda.amp.autocast
#     all_preds, all_labels = [], []
#     total_correct1, total_correct5, total_count = 0.0, 0.0, 0


#     all_gating_weights = []
#     all_class_ids = []


#     with torch.no_grad():
#         n_class = text_features.size(0)

#         for i, (image, class_id) in enumerate(val_loader):
#             batch_size = class_id.numel()
#             num_crop = test_crops * test_clips
#             class_id = class_id.to(device, non_blocking=True)
#             n_seg = config.data.num_segments

#             # reshape image batch: [batch × num_crop, num_segments, 3, H, W]
#             image = image.view((-1, n_seg, 3) + image.size()[-2:])
#             b, t, c, h, w = image.size()
#             actual_num_crop = b // batch_size
#             image_input = image.to(device, non_blocking=True).view(-1, c, h, w)

#             # forward
#             with autocast_ctx():
#                 if distributed:
#                     image_features, _ = model.module(image_input)
#                 else:
#                     image_features, _ = model(image_input)
#                 # B, T, D = image_features.size()
#                 # features = image_features.view(B, T, -1).permute(1, 0, 2)
#                 # curves, lengths = video_head.plot_experts_global_temporal_attention(
#                 #     features_LBD=features,

#                 #     save_dir="expert_global_curves",




#                 # )
#                 # print("curves", curves)

#                 vis_emb_batch = image_features.mean(dim=1)  


#                 vis_emb_batch = vis_emb_batch.view(batch_size, actual_num_crop, -1).mean(dim=1)
#                 # print('vis_emb_batch shape ', vis_emb_batch.shape)

#                 # all_vis_embs.append(vis_emb_batch.detach().cpu())
#                 logits, projected_features, gating_weights, div_loss, extra_loss = video_head(
#                     image_features, b, t, text_embeds=text_features
#                 )
#                 # logits, projected_features, gating_weights, div_loss, extra_loss = video_head(
#                 #     image_features, b, t
#                 # )
#                 # print("projected_features shape: ", projected_features.shape)
#                 all_vis_embs.append(projected_features.detach().cpu())

#             all_gating_weights.append(gating_weights.detach().cpu())
#             all_class_ids.append(class_id.detach().cpu())

#             # multi-crop averaging logits
#             num_classes = logits.size(-1)
#             logits = logits.view(batch_size, actual_num_crop, num_classes).mean(dim=1)
#             similarity = F.softmax(logits, dim=-1)

#             # top-k accuracy
#             maxk = max((1, 5))
#             _, pred = similarity.topk(maxk, 1, True, True)
#             pred = pred.t()
#             correct = pred.eq(class_id.view(1, -1).expand_as(pred))
#             correct1 = correct[:1].reshape(-1).float().sum().item()
#             correct5 = correct[:5].reshape(-1).float().sum().item()

#             bs = class_id.size(0)
#             total_correct1 += correct1
#             total_correct5 += correct5
#             total_count += bs
#             all_preds.append(pred[0].cpu())
#             all_labels.append(class_id.cpu())

#             if rank == 0 and (i % config.logging.print_freq == 0):
#                 local_top1 = 100.0 * correct1 / bs
#                 local_top5 = 100.0 * correct5 / bs
#                 runtime = float(time.time() - proc_start_time) / (i + 1) / bs
#                 print(f"Test: [{i}/{len(val_loader)}], average {runtime:.4f}s/video, "
#                       f"Prec@1 {local_top1:.3f}, Prec@5 {local_top5:.3f}")


#     if distributed and dist.is_available() and dist.is_initialized():
#         t1, t5, tc = map(lambda x: torch.tensor([x], device=device), [total_correct1, total_correct5, total_count])
#         dist.all_reduce(t1, op=dist.ReduceOp.SUM)
#         dist.all_reduce(t5, op=dist.ReduceOp.SUM)
#         dist.all_reduce(tc, op=dist.ReduceOp.SUM)
#         total_correct1, total_correct5, total_count = t1.item(), t5.item(), int(tc.item())


#     top1_avg = 100.0 * total_correct1 / max(1, total_count)
#     top5_avg = 100.0 * total_correct5 / max(1, total_count)

#     if rank == 0:
#         print(f"\n=== Full-Class Evaluation ===")
#         print(f"Prec@1 {top1_avg:.3f}, Prec@5 {top5_avg:.3f}\n")

#         # =========================

#         # =========================
#         all_preds_tensor = torch.cat(all_preds)
#         all_labels_tensor = torch.cat(all_labels)
#         cm = confusion_matrix(all_labels_tensor.numpy(), all_preds_tensor.numpy())
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(cm, ax=ax, cmap="Blues", annot=False, fmt='d')
#         ax.set_xlabel("Predicted label")
#         ax.set_ylabel("True label")
#         ax.set_title("Confusion Matrix")
#         plt.tight_layout()
#         plt.savefig("confusion_matrix.png", dpi=400)
#         plt.close()
#         print("[INFO] Confusion matrix saved to confusion_matrix.png")

#         # =========================

#         # =========================
#         all_gating_weights = torch.cat(all_gating_weights, dim=0)
#         all_class_ids = torch.cat(all_class_ids, dim=0)
#         # visualize_expert_per_class(
#         #     all_gating_weights, all_class_ids,
#         #     save_path="expert_usage_per_class.png",
#         #     top_k_classes=30
#         # )
#         print("[INFO] Saved class-wise expert usage heatmap.\n")
#         classes = visualize_expert_correct_wrong_per_class(
#             gating_weights=all_gating_weights,
#             preds=torch.cat(all_preds, dim=0),
#             labels=torch.cat(all_labels, dim=0),
#             save_dir="expert_usage_correct_wrong"
#         )
#         # =========================

#         # =========================
#         print("=== Half-Class Evaluation ===")

#         acc_split, acc5_split = multi_split_test(
#             vis_embs=all_vis_embs_tensor,

#             true_label=all_labels_tensor
#         )
#         print("Top1 of 10 splits:", np.round(acc_split, 3))
#         print("Top5 of 10 splits:", np.round(acc5_split, 3))
#         print(f"Half-Class Mean Top1: {acc_split.mean():.3f}, Mean Top5: {acc5_split.mean():.3f}\n")

#     return top1_avg, top5_avg, classes



def validate_train(val_loader, device, model, video_head, config, text_features,
             test_crops, test_clips, distributed=False, rank=0, classes=None):
    """Distributed validation with multi-crop/clip averaging + Half-Class Evaluation"""
    model.eval()
    video_head.eval()
    proc_start_time = time.time()
    autocast_ctx = torch.cuda.amp.autocast
    all_preds, all_labels = [], []
    total_correct1, total_correct5, total_count = 0.0, 0.0, 0

    # Collect gating_weights, class_id, and vis_embs
    all_gating_weights = []
    all_class_ids = []
    all_vis_embs = []  

    with torch.no_grad():
        n_class = text_features.size(0)

        for i, (image, class_id) in enumerate(val_loader):
            batch_size = class_id.numel()
            num_crop = test_crops * test_clips
            class_id = class_id.to(device, non_blocking=True)
            n_seg = config.data.num_segments

            # reshape image batch: [batch × num_crop, num_segments, 3, H, W]
            image = image.view((-1, n_seg, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            actual_num_crop = b // batch_size
            image_input = image.to(device, non_blocking=True).view(-1, c, h, w)

            # forward
            with autocast_ctx():
                if distributed:
                    image_features, _ = model.module(image_input)
                else:
                    image_features, _ = model(image_input)
                # B, T, D = image_features.size()
                # features = image_features.view(B, T, -1).permute(1, 0, 2)
                # curves, lengths = video_head.plot_experts_global_temporal_attention(
                #     features_LBD=features,
                
                #     save_dir="expert_global_curves",
                
                
                
                
                # )
                # print("curves", curves)
                
                vis_emb_batch = image_features.mean(dim=1)  

                
                vis_emb_batch = vis_emb_batch.view(batch_size, actual_num_crop, -1).mean(dim=1)
                # print('vis_emb_batch shape ', vis_emb_batch.shape)
                
                # all_vis_embs.append(vis_emb_batch.detach().cpu())
                logits, projected_features, gating_weights, div_loss, extra_loss = video_head(
                    image_features, b, t, text_embeds=text_features
                )
                # print("projected_features shape: ", projected_features.shape)
                all_vis_embs.append(projected_features.detach().cpu())
            
            all_gating_weights.append(gating_weights.detach().cpu())
            all_class_ids.append(class_id.detach().cpu())

            # multi-crop averaging logits
            num_classes = logits.size(-1)
            logits = logits.view(batch_size, actual_num_crop, num_classes).mean(dim=1)
            similarity = F.softmax(logits, dim=-1)

            # top-k accuracy
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
            all_preds.append(pred[0].cpu())
            all_labels.append(class_id.cpu())

            if rank == 0 and (i % config.logging.print_freq == 0):
                local_top1 = 100.0 * correct1 / bs
                local_top5 = 100.0 * correct5 / bs
                runtime = float(time.time() - proc_start_time) / (i + 1) / bs
                print(f"Test: [{i}/{len(val_loader)}], average {runtime:.4f}s/video, "
                      f"Prec@1 {local_top1:.3f}, Prec@5 {local_top5:.3f}")

    # Aggregate distributed results
    if distributed and dist.is_available() and dist.is_initialized():
        t1, t5, tc = map(lambda x: torch.tensor([x], device=device), [total_correct1, total_correct5, total_count])
        dist.all_reduce(t1, op=dist.ReduceOp.SUM)
        dist.all_reduce(t5, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
        total_correct1, total_correct5, total_count = t1.item(), t5.item(), int(tc.item())

    # Compute the final average top1/top5
    top1_avg = 100.0 * total_correct1 / max(1, total_count)
    top5_avg = 100.0 * total_correct5 / max(1, total_count)

    if rank == 0:
        print(f"\n=== Full-Class Evaluation ===")
        print(f"Prec@1 {top1_avg:.3f}, Prec@5 {top5_avg:.3f}\n")

        # =========================
        # 1. Confusion matrix
        # =========================
        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)
        cm = confusion_matrix(all_labels_tensor.numpy(), all_preds_tensor.numpy())
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=False, fmt='d')
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=400)
        plt.close()
        print("[INFO] Confusion matrix saved to confusion_matrix.png")

        # =========================
        # 2. Expert-usage heatmap
        # =========================
        all_gating_weights = torch.cat(all_gating_weights, dim=0)
        all_class_ids = torch.cat(all_class_ids, dim=0)
        if classes is not None:
            visualize_expert_per_class(
                all_gating_weights, all_class_ids,
                save_path="expert_usage_per_class.png",
                top_k_classes=len(classes),
                valid_classes=classes
            )
        print("[INFO] Saved class-wise expert usage heatmap.\n")
        # classes = visualize_expert_correct_wrong_per_class(
        #     gating_weights=all_gating_weights,
        #     preds=torch.cat(all_preds, dim=0),
        #     labels=torch.cat(all_labels, dim=0),
        #     save_dir="expert_usage_correct_wrong"
        # )
        # =========================
        # 3. Half-class evaluation
        # =========================
        print("=== Half-Class Evaluation ===")
        all_vis_embs_tensor = torch.cat(all_vis_embs, dim=0).float()      
        acc_split, acc5_split = multi_split_test(
            vis_embs=all_vis_embs_tensor,
            text_embs=text_features.cpu().float(),                        
            true_label=all_labels_tensor
        )
        print("Top1 of 10 splits:", np.round(acc_split, 3))
        print("Top5 of 10 splits:", np.round(acc5_split, 3))
        print(f"Half-Class Mean Top1: {acc_split.mean():.3f}, Mean Top5: {acc5_split.mean():.3f}\n")

    return top1_avg, top5_avg






# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return tensor.cpu()
        
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output.cpu()

def compute_accuracy(vis_emb, text_emb, label):
    
    if vis_emb.dtype != text_emb.dtype:
        vis_emb = vis_emb.float()
        text_emb = text_emb.float()

    n_class = text_emb.size(0)  # num_classes
    n_samples = vis_emb.size(0) # num_videos

    
    similarity = (100.0 * vis_emb @ text_emb.T)
    similarity = similarity.softmax(dim=-1)

    # top-k accuracy
    prec = accuracy(similarity, label, topk=(1, 5))
    return prec[0], prec[1]

 
 
def multi_split_test(vis_embs, text_embs, true_label):
    full_acc1, full_acc5 = compute_accuracy(vis_embs, text_embs, true_label)
    print('-----Full-classes Evaluation------')
    print('Overall Top1 {:.03f}% Top5 {:.03f}%'.format(full_acc1.item(), full_acc5.item()))
 
    # Calculate accuracy per split
    # Only when the model has been trained on a different dataset
    true_label = true_label.numpy()
    accuracy_split, accuracy_split_top5 = np.zeros(10), np.zeros(10)
    for split in range(len(accuracy_split)):
        np.random.seed(split)
        sel_classes = np.random.permutation(len(text_embs))[:len(text_embs) // 2]  # [50, ]
        sel = [l in sel_classes for l in true_label]    # len = 10000 [<num_video]
        subclasses = np.unique(true_label[sel])         # [num_class//2 ]
        tl = np.array([int(np.where(l == subclasses)[0]) for l in true_label[sel]])
        tl = torch.from_numpy(tl)
        acc, acc5 = compute_accuracy(vis_embs[sel], text_embs[subclasses], tl)
        accuracy_split[split] = acc
        accuracy_split_top5[split] = acc5
    
    return accuracy_split, accuracy_split_top5


def visualize_expert_per_class(
    gating_weights,
    class_id,
    label_csv="/home/linrui/moe/TimeMoE/lists/ucf101/ucf_labels.csv",
    save_path="expert_usage_per_class.png",
    top_k_classes=101,
    valid_classes=None
):

    
    label_map = pd.read_csv(label_csv)
    id_to_name = dict(zip(label_map["id"], label_map["name"]))

    
    gw = gating_weights.detach().cpu()
    class_id = class_id.cpu()
    if valid_classes is not None:
        mask = torch.tensor([c in valid_classes for c in class_id])
        gw = gw[mask]
        class_id = class_id[mask]
    num_experts = gw.size(1)
    gw_softmax = torch.nn.functional.softmax(gw, dim=1)  

    df = pd.DataFrame(gw_softmax.numpy(), columns=[f"Expert {i}" for i in range(num_experts)])
    df["class"] = class_id.numpy()

    class_usage = df.groupby("class").mean()
    class_usage["class_name"] = class_usage.index.map(lambda x: id_to_name.get(x, f"id_{x}"))
    class_usage = class_usage.sort_index().set_index("class_name").iloc[:top_k_classes, :]
    class_usage_T = class_usage.T
    min_val = class_usage_T.min().min()  
    max_val = class_usage_T.max().max()

    
    # plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(figsize=(max(10, len(valid_classes)*0.4),
                        max(5, num_experts*0.4)))
    sns.heatmap(class_usage_T, cmap="Blues",ax=ax, square=True, vmin=min_val, vmax=max_val, cbar=False) # cmap="viridis"
    ax.set_title("Expert Usage On Training Set")
    
    ax.set_xticks([])         
    # ax.set_ylabel("Expert Index")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    ax.set_aspect('equal')

    plt.tight_layout()

    
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=800)
    plt.close()
    print(f"[INFO] Saved class-level expert distribution figure: {save_path}")

import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

def visualize_expert_correct_wrong_per_class(
    gating_weights,
    preds,
    labels,
    save_dir=".",
    top_k_classes=None,
    min_samples=3
):
    """
    Plot expert-usage heatmaps per class for test samples.
    One heatmap is rendered for correct samples and one for incorrect samples.

    Args:
        gating_weights: Tensor [N, num_experts]
        preds: Tensor [N]
        labels: Tensor [N]
        save_dir: Output directory
        top_k_classes: Optional limit on the number of classes to display
        min_samples: Minimum samples per class before filtering
    """
    os.makedirs(save_dir, exist_ok=True)

    gating_weights = gating_weights.detach().cpu()
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    num_experts = gating_weights.size(1)
    num_classes = int(labels.max().item() + 1)

    
    gw_softmax = F.softmax(gating_weights, dim=1)

    
    correct_mask = (preds == labels)
    wrong_mask = ~correct_mask

    
    expert_usage_correct = []
    expert_usage_wrong = []
    valid_classes = []

    for cls_id in range(num_classes):
        
        mask_cls = (labels == cls_id)
        if mask_cls.sum().item() == 0:
            continue

        mask_correct_cls = mask_cls & correct_mask
        mask_wrong_cls = mask_cls & wrong_mask

        n_correct = mask_correct_cls.sum().item()
        n_wrong = mask_wrong_cls.sum().item()

        
        if n_correct < min_samples or n_wrong < min_samples:
            continue

        mean_correct = gw_softmax[mask_correct_cls].mean(dim=0)
        mean_wrong = gw_softmax[mask_wrong_cls].mean(dim=0)

        expert_usage_correct.append(mean_correct.numpy())
        expert_usage_wrong.append(mean_wrong.numpy())
        valid_classes.append(cls_id)

    if len(valid_classes) == 0:
        print("[WARN] No class has enough correct and incorrect samples to draw the heatmaps.")
        return

    
    df_correct = pd.DataFrame(expert_usage_correct,
                              index=valid_classes,
                              columns=[f"Expert {i}" for i in range(num_experts)])
    df_wrong = pd.DataFrame(expert_usage_wrong,
                            index=valid_classes,
                            columns=[f"Expert {i}" for i in range(num_experts)])

    
    min_val = min(df_correct.min().min(), df_wrong.min().min())
    max_val = max(df_correct.max().max(), df_wrong.max().max())

    
    plt.rcParams['font.size'] = 16

    plt.figure(figsize=(max(10, len(valid_classes)*0.4),
                        max(5, num_experts*0.4)))
    sns.heatmap(df_correct.T, cmap="Blues", cbar=False, square=True, vmin=min_val, vmax=max_val)
    plt.title("Expert Usage (Test Correct Samples)")
    # plt.xlabel("Class ID")
    plt.xticks(rotation=0, ha="right")
    plt.yticks(rotation=0)
    # plt.set_xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "expert_usage_correct_per_class.png"), dpi=800)
    plt.close()

    
    plt.figure(figsize=(max(10, len(valid_classes)*0.4),
                        max(5, num_experts*0.4)))
    sns.heatmap(df_wrong.T, cmap="Blues", cbar=False, square=True, vmin=min_val, vmax=max_val)
    plt.title("Expert Usage (Test Wrong Samples)")
    plt.xlabel("Class ID")
    # plt.ylabel("Expert Index")
    plt.xticks(rotation=0, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "expert_usage_wrong_per_class.png"), dpi=800)
    plt.close()

    print(f"[INFO] Saved correct-sample heatmap → {os.path.join(save_dir, 'expert_usage_correct_per_class.png')}")
    print(f"[INFO] Saved wrong-sample  heatmap → {os.path.join(save_dir, 'expert_usage_wrong_per_class.png')}")

    return valid_classes

if __name__ == '__main__':
    args = get_parser()
    # args.distributed = False
    main(args)

