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
from VidPrism.modules.videomoe import VidPrism, VideoCLIP
from modules.text_prompt import text_prompt, text_prompt_ensemble

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



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

    prec1, prec5 = validate(
        val_loader, device,
        model, video_head, config, cls_feature, args.test_crops, args.test_clips, args.distributed, rank)
    return


def validate(val_loader, device, model, video_head, config, text_features, test_crops, test_clips, distributed=False, rank=0):
    """Distributed validation with multi‑crop/clip averaging."""
    model.eval()
    video_head.eval()
    proc_start_time = time.time()
    autocast_ctx = torch.cuda.amp.autocast
    all_preds = []
    all_labels = []
    total_correct1, total_correct5, total_count = 0.0, 0.0, 0

    with torch.no_grad():
        n_class = text_features.size(0)

        for i, (image, class_id) in enumerate(val_loader):
            batch_size = class_id.numel()
            num_crop = test_crops * test_clips        # nominal crop×clip
            class_id = class_id.to(device, non_blocking=True)
            n_seg = config.data.num_segments

            # reshape image batch: [batch × num_crop, num_segments, 3, H, W]
            image = image.view((-1, n_seg, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            actual_num_crop = b // batch_size
            assert b == batch_size * actual_num_crop, \
                f"b({b}) != batch_size({batch_size}) * actual_num_crop({actual_num_crop})"

            image_input = image.to(device, non_blocking=True).view(-1, c, h, w)

            # forward
            with autocast_ctx():
                if distributed:
                    image_features, _ = model.module(image_input)
                else:
                    image_features, _ = model(image_input)
                logits, gating_weights, div_loss, extra_loss = video_head(image_features, b, t, text_embeds=text_features)
                # logits: [batch_size * actual_num_crop, num_classes]

            # ----- multi‑crop averaging (in logits space) -----
            num_classes = logits.size(-1)
            logits = logits.view(batch_size, actual_num_crop, num_classes)
            logits = logits.mean(dim=1)                   # [batch_size, num_classes]

            # convert to probabilities
            similarity = F.softmax(logits, dim=-1)        # [batch_size, num_classes]

            # ----- compute Top‑1 / Top‑5 counts -----
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
            # ----- logging -----
            if rank == 0 and (i % config.logging.print_freq == 0):
                local_top1 = 100.0 * correct1 / bs
                local_top5 = 100.0 * correct5 / bs
                runtime = float(time.time() - proc_start_time) / (i + 1) / bs
                print(
                    f"Test: [{i}/{len(val_loader)}], average {runtime:.4f}s/video, "
                    f"Prec@1 {local_top1:.3f}, Prec@5 {local_top5:.3f}"
                )

    # ----- distributed reduce -----
    if distributed and dist.is_available() and dist.is_initialized():
        t1 = torch.tensor([total_correct1], device=device)
        t5 = torch.tensor([total_correct5], device=device)
        tc = torch.tensor([total_count], device=device)
        dist.all_reduce(t1, op=dist.ReduceOp.SUM)
        dist.all_reduce(t5, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
        total_correct1, total_correct5, total_count = t1.item(), t5.item(), int(tc.item())

    top1_avg = 100.0 * total_correct1 / max(1, total_count)
    top5_avg = 100.0 * total_correct5 / max(1, total_count)

    if rank == 0:
        print(f"Testing Results: Prec@1 {top1_avg:.3f}, Prec@5 {top5_avg:.3f}")
        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)
        cm = confusion_matrix(all_labels_tensor.numpy(), all_preds_tensor.numpy())
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=False, fmt='d')
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        print("[INFO] Confusion matrix saved to confusion_matrix.png")

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
    n_class = len(text_emb) # int: num_class
    n_samples = len(vis_emb) # int: num_video
    similarity=(100.0 * vis_emb @ text_emb.T) # [num_video, num_crop, num_class]

    # fuse then normalize
    similarity=similarity.mean(dim = 1, keepdim = False)  # b 101 [num_videos, num_classes]
    similarity=similarity.view(n_samples, n_class).softmax(dim = -1)
    
    # similarity: [num_videos, num_classes]
    prec=accuracy(similarity, label, topk = (1, 5))
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


if __name__ == '__main__':
    args = get_parser()
    # args.distributed = False
    main(args)

