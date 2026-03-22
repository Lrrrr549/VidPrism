import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR, WarmupCosineAnnealingStepLR, WarmupMultiStepStepLR

def _optimizer(config, model, video_head):
    if config.solver.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()},  
         {'params': video_head.parameters(), 'lr': config.solver.lr}],
                               lr=config.solver.lr * config.solver.clip_ratio, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print('Adam')
    elif config.solver.optim == 'sgd':

        optimizer = optim.SGD([{'params': model.parameters()},  
         {'params': video_head.parameters(), 'lr': config.solver.lr}],
                              config.solver.lr * config.solver.clip_ratio,
                              momentum=config.solver.momentum,
                              weight_decay=config.solver.weight_decay)
        print('SGD')
    elif config.solver.optim == 'adamw':
        vision_params = []
        text_params = []
        for name, param in model.named_parameters():
            if 'visual.' in name:
                vision_params.append(param)
            else:
                text_params.append(param)       
            # if 'visual.' not in name and 'logit_scale' not in name:
            #     param.requires_grad = False 

        # print('[INFO] number of visual parameters:', len(vision_params), flush=True)
        # print('[INFO] number of textual parameters:', len(text_params), flush=True)
        optimizer = optim.AdamW([{'params': model.parameters(), 'lr': config.solver.lr * config.solver.clip_ratio},
                                 {'params': video_head.parameters(), 'lr': config.solver.lr}],
                                betas=(0.9, 0.999), lr=config.solver.lr, eps=1e-8,
                                weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.solver.optim))
    return optimizer


def _lr_scheduler(config, optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler


def _lr_scheduler_step(config, optimizer, steps_per_epoch):
    # 从配置文件读取warmup设置
    warmup_epochs = config.solver.epochs
    warmup_steps = config.solver.steps
    warmup_lr = config.solver.lr

    # 计算以 step 为单位的 warmup 总步数
    if warmup_epochs > 0:
        warmup_total_steps = int(warmup_epochs * steps_per_epoch)
    else:
        warmup_total_steps = warmup_steps
        
    print(f"[LR_Scheduler] Using step-based scheduler. Total warmup steps: {warmup_total_steps}")


    if config.solver.type == 'cosine':
        total_steps = config.solver.epochs * steps_per_epoch
        
        # --- 使用新的、基于Step的调度器 ---
        lr_scheduler = WarmupCosineAnnealingStepLR(
            optimizer,
            total_steps,
            warmup_steps=warmup_total_steps,
            warmup_lrs=warmup_lr
        )
        
    elif config.solver.type == 'multistep':
        milestones_in_epochs = config.solver.lr_decay_step
        milestones_in_steps = [int(m * steps_per_epoch) for m in milestones_in_epochs]

        # --- 使用新的、基于Step的调度器 ---
        lr_scheduler = WarmupMultiStepStepLR(
            optimizer,
            milestones_in_steps,
            warmup_steps=warmup_total_steps,
            warmup_lrs=warmup_lr
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler