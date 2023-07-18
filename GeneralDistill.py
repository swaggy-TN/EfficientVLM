# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import argparse
import os
import sys

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import math
from pathlib import Path
import torch
from torch.nn import MSELoss,KLDivLoss
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.data.dataloader import T

from models.model_pretrain import XVLM


import utils
from dataset import create_dataset
from scheduler import create_scheduler
from optim import create_optimizer

from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hcopy,hexists
from accelerators.apex_ddp_accelerator import ApexDDPAccelerator


def reinit_scheduler_properties_mysched(optimizer: Optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    issue: https://github.com/pytorch/pytorch/issues/27595
    issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
    """
    args = cfg

    if scheduler.optimizer == optimizer:
        # from transformers import get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        scheduler.__init__(optimizer, lr_lambda, last_epoch=-1)


def get_kd_loss(student_reps=None,teacher_reps=None,is_attn=False,loss=None,device='cuda',is_img=False):
    kd_loss = 0
    if is_attn:
        for student_att, teacher_att in zip(student_reps, teacher_reps):
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                        student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                        teacher_att)
            
            kd_loss += loss(student_att, teacher_att) * student_att.shape[-1]
    elif is_img:
        layer = 0
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            #drop the loss of the last layer
            if layer == 6:
                pass
            else:
                kd_loss += loss(student_rep, teacher_rep)
            layer += 1
    else:
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            kd_loss += loss(student_rep, teacher_rep)
    return kd_loss

def soft_cross_entropy(predicts, targets):
    kl_loss = KLDivLoss(reduction='batchmean')
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)

    return kl_loss(student_likelihood.view(-1,predicts.shape[-1]),targets_prob.view(-1,targets.shape[-1]))

def get_cor_teacher(teacher_reps,student_reps,is_attn=False):
    teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]
    teacher_layer_num = len(teacher_reps)
    student_layer_num = len(student_reps)
    if is_attn:
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)
        new_teacher_reps = [teacher_reps[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]
    else:
        assert (teacher_layer_num-1) % (student_layer_num-1) == 0
        layers_per_block = int((teacher_layer_num-1) / (student_layer_num-1))
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num)]
    return new_teacher_reps


def train(teacher_model,student_model, general_loader, region_loader, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer):
    with torch.no_grad():
        teacher_model.eval()
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_img_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_text_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cross_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_small', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_itm_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_mlm_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_img_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_text_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_cross_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_bbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_giou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('region_loss_small', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_large', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train step: [{}]'.format(start_epoch)

    print_freq = 50

    world_size = utils.get_world_size()
    step_per_epoch = math.ceil(config['train_dataset_size']/(config['batch_size']*world_size))
    assert step_per_epoch > 1
    global_step = step_per_epoch * start_epoch + 1  

    subarea_iter = iter(region_loader)

    output_attentions = config['output_attentions']
    output_hidden_states = config['output_hidden_states']
    assert output_attentions == output_hidden_states
    
    if output_hidden_states:
        student_model.train()
    else:
        student_model.eval()

    for i, batch in enumerate(metric_logger.log_every(general_loader, print_freq, header, step_per_epoch, epoch_info)):
        if random.random() < config['regions']['iter_perc']:
            try:
                region_batch = next(subarea_iter)
            except StopIteration:
                subarea_iter = iter(region_loader)
                region_batch = next(subarea_iter)

            image, region_batch = region_batch[0].to(device, non_blocking=True), [
                t.to(device) if t is not None else None for t in region_batch[1:]]

            idx_to_group_img, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
                image_atts, target_bbox, is_image = region_batch

            if config['calc_image_bbox_loss']:
                is_image = None

            optimizer.zero_grad()

            student_outputs = student_model(image, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                      image_atts=image_atts, idx_to_group_img=idx_to_group_img, target_bbox=target_bbox, is_image=is_image, ret_bbox_loss=True,
                      output_attentions=output_attentions,output_hidden_states=output_hidden_states)
            with torch.no_grad():
                teacher_outputs = teacher_model(image, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                      image_atts=image_atts, idx_to_group_img=idx_to_group_img, target_bbox=target_bbox, is_image=is_image, ret_bbox_loss=True,
                      output_attentions=output_attentions,output_hidden_states=output_hidden_states)
            
            student_hidden = student_outputs['hidden_dict']
            teacher_hidden = teacher_outputs['hidden_dict']
            student_attentions = student_outputs['attention_dict']
            teacher_attentions = teacher_outputs['attention_dict']
            student_logits = student_outputs['logits_dict']
            teacher_logits = teacher_outputs['logits_dict']
            mse_loss = MSELoss()

            
            #text kd
            student_text_hidden = student_hidden['text_hidden_states']
            teacher_text_hidden = teacher_hidden['text_hidden_states']
            teacher_text_hidden = get_cor_teacher(teacher_text_hidden,student_text_hidden)
            student_text_attn = student_attentions['text_attentions']
            teacher_text_attn = teacher_attentions['text_attentions']
            teacher_text_attn = get_cor_teacher(teacher_text_attn,student_text_attn,is_attn=True)
            text_hidden_loss = get_kd_loss(student_text_hidden,teacher_text_hidden,False,mse_loss,device)
            text_attention_loss = get_kd_loss(student_text_attn,teacher_text_attn,True,mse_loss,device)

            #image kd
            student_image_hidden = student_hidden['image_hidden_states']
            teacher_image_hidden = teacher_hidden['image_hidden_states']
            teacher_image_hidden = get_cor_teacher(teacher_image_hidden,student_image_hidden)
            student_image_attn = student_attentions['image_attentions']
            teacher_image_attn = teacher_attentions['image_attentions']
            teacher_image_attn = get_cor_teacher(teacher_image_attn,student_image_attn,is_attn=True)
            image_hidden_loss = get_kd_loss(student_image_hidden,teacher_image_hidden,False,mse_loss,device,is_img=True)
            image_attention_loss = get_kd_loss(student_image_attn,teacher_image_attn,True,mse_loss,device)
            
            #cross kd
            #itm positive examples
            student_pos_hidden = student_hidden['itm_pos_hidden_states']
            teacher_pos_hidden = teacher_hidden['itm_pos_hidden_states']
            teacher_pos_hidden = get_cor_teacher(teacher_pos_hidden,student_pos_hidden)
            student_pos_attn = student_attentions['itm_pos_attentions']
            teacher_pos_attn = teacher_attentions['itm_pos_attentions']
            teacher_pos_attn = get_cor_teacher(teacher_pos_attn,student_pos_attn,is_attn=True)
            itm_pos_hidden_loss = get_kd_loss(student_pos_hidden,teacher_pos_hidden,False,mse_loss,device)
            itm_pos_attn_loss = get_kd_loss(student_pos_attn,teacher_pos_attn,True,mse_loss,device)
            #itm negative examples
            student_neg_hidden = student_hidden['itm_neg_hidden_states']
            teacher_neg_hidden = teacher_hidden['itm_neg_hidden_states']
            teacher_neg_hidden = get_cor_teacher(teacher_neg_hidden,student_neg_hidden)
            student_neg_attn = student_attentions['itm_neg_attentions']
            teacher_neg_attn = teacher_attentions['itm_neg_attentions']
            teacher_neg_attn = get_cor_teacher(teacher_neg_attn,student_neg_attn,is_attn=True)
            itm_neg_hidden_loss = get_kd_loss(student_neg_hidden,teacher_neg_hidden,False,mse_loss,device)
            itm_neg_attn_loss = get_kd_loss(student_neg_attn,teacher_neg_attn,True,mse_loss,device)
            #mlm 
            student_mlm_hidden = student_hidden['mlm_hidden_states']
            teacher_mlm_hidden = teacher_hidden['mlm_hidden_states']
            teacher_mlm_hidden = get_cor_teacher(teacher_mlm_hidden,student_mlm_hidden)
            student_mlm_attn = student_attentions['mlm_attentions']
            teacher_mlm_attn = teacher_attentions['mlm_attentions']
            teacher_mlm_attn = get_cor_teacher(teacher_mlm_attn,student_mlm_attn,is_attn=True)
            mlm_hidden_loss = get_kd_loss(student_mlm_hidden,teacher_mlm_hidden,False,mse_loss,device)
            mlm_attn_loss = get_kd_loss(student_mlm_attn,teacher_mlm_attn,True,mse_loss,device)
            
            #logits kd loss
            #mlm head
            student_mlm_logits = student_logits['mlm_logits']
            teacher_mlm_logits= teacher_logits['mlm_logits']
            mlm_logits_loss = soft_cross_entropy(student_mlm_logits/args.temperature, teacher_mlm_logits/args.temperature)
            #itm head
            student_itm_logits = student_logits['itm_head_logits']
            teacher_itm_logits = teacher_logits['itm_head_logits']
            itm_logits_loss = soft_cross_entropy(student_itm_logits/args.temperature, teacher_itm_logits/args.temperature)
            
            loss = student_outputs['loss']
            loss_text_kd =  text_attention_loss + text_hidden_loss
            loss_img_kd = image_attention_loss + 0.1 * image_hidden_loss
            loss_cross_kd =  itm_neg_attn_loss + itm_neg_hidden_loss + itm_pos_attn_loss + itm_pos_hidden_loss + mlm_attn_loss + mlm_hidden_loss
            loss_kd = itm_logits_loss +  mlm_logits_loss + loss_text_kd + loss_img_kd + loss_cross_kd

            loss_small = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm'] + loss['loss_bbox'] + loss['loss_giou']
            
            loss_in_total = 0.6*loss_small + 0.4*loss_kd
            accelerator.backward_step(loss_in_total, optimizer)

            accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
            if accelerator_clip_grad_norm > 0:
                accelerator.optimizer_step(optimizer, student_model, accelerator_clip_grad_norm)
            optimizer.step()

            metric_logger.update(region_loss_bbox=loss['loss_bbox'].item())
            metric_logger.update(region_loss_giou=loss['loss_giou'].item())
            metric_logger.update(region_loss_itc=loss['loss_itc'].item())
            metric_logger.update(region_loss_itm=loss['loss_itm'].item())
            metric_logger.update(region_loss_itm_kd=itm_logits_loss.item())
            metric_logger.update(region_loss_mlm=loss['loss_mlm'].item())
            metric_logger.update(region_loss_mlm_kd=mlm_logits_loss.item())
            metric_logger.update(region_loss_text_kd=loss_text_kd.item())
            metric_logger.update(region_loss_img_kd=loss_img_kd.item())
            metric_logger.update(region_loss_cross_kd=loss_cross_kd.item())
            metric_logger.update(region_loss_kd=loss_kd.item())
            metric_logger.update(region_loss_small=loss_small.item())

        else:
            #fix it
            metric_logger.update(loss_bbox=0.5)
            metric_logger.update(loss_giou=0.5)

        image, batch = batch[0].to(device, non_blocking=True), [t.to(device) if t is not None else None for t in batch[1:]]
        text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = batch

        optimizer.zero_grad()

        if output_attentions:
            student_outputs = student_model(image, text_ids, text_atts, text_ids_masked=text_ids_masked,
                                            masked_pos=masked_pos, masked_ids=masked_ids,
                                            output_attentions=output_attentions,output_hidden_states=output_hidden_states)
            with torch.no_grad():
                teacher_outputs = teacher_model(image, text_ids, text_atts, text_ids_masked=text_ids_masked,
                                        masked_pos=masked_pos, masked_ids=masked_ids,
                                        output_attentions=output_attentions,output_hidden_states=output_hidden_states)
                
        student_hidden = student_outputs['hidden_dict']
        teacher_hidden = teacher_outputs['hidden_dict']
        student_attentions = student_outputs['attention_dict']
        teacher_attentions = teacher_outputs['attention_dict']
        student_logits = student_outputs['logits_dict']
        teacher_logits = teacher_outputs['logits_dict']
        mse_loss = MSELoss()

        
        #text kd
        student_text_hidden = student_hidden['text_hidden_states']
        teacher_text_hidden = teacher_hidden['text_hidden_states']
        teacher_text_hidden = get_cor_teacher(teacher_text_hidden,student_text_hidden)
        student_text_attn = student_attentions['text_attentions']
        teacher_text_attn = teacher_attentions['text_attentions']
        teacher_text_attn = get_cor_teacher(teacher_text_attn,student_text_attn,is_attn=True)
        text_hidden_loss = get_kd_loss(student_text_hidden,teacher_text_hidden,False,mse_loss,device)
        text_attention_loss = get_kd_loss(student_text_attn,teacher_text_attn,True,mse_loss,device)

        #image kd
        student_image_hidden = student_hidden['image_hidden_states']
        teacher_image_hidden = teacher_hidden['image_hidden_states']
        teacher_image_hidden = get_cor_teacher(teacher_image_hidden,student_image_hidden)
        student_image_attn = student_attentions['image_attentions']
        teacher_image_attn = teacher_attentions['image_attentions']
        teacher_image_attn = get_cor_teacher(teacher_image_attn,student_image_attn,is_attn=True)
        image_hidden_loss = get_kd_loss(student_image_hidden,teacher_image_hidden,False,mse_loss,device,is_img=True)
        image_attention_loss = get_kd_loss(student_image_attn,teacher_image_attn,True,mse_loss,device)
        
        #cross kd
        #itm positive examples
        student_pos_hidden = student_hidden['itm_pos_hidden_states']
        teacher_pos_hidden = teacher_hidden['itm_pos_hidden_states']
        teacher_pos_hidden = get_cor_teacher(teacher_pos_hidden,student_pos_hidden)
        student_pos_attn = student_attentions['itm_pos_attentions']
        teacher_pos_attn = teacher_attentions['itm_pos_attentions']
        teacher_pos_attn = get_cor_teacher(teacher_pos_attn,student_pos_attn,is_attn=True)
        itm_pos_hidden_loss = get_kd_loss(student_pos_hidden,teacher_pos_hidden,False,mse_loss,device)
        itm_pos_attn_loss = get_kd_loss(student_pos_attn,teacher_pos_attn,True,mse_loss,device)
        #itm negative examples
        student_neg_hidden = student_hidden['itm_neg_hidden_states']
        teacher_neg_hidden = teacher_hidden['itm_neg_hidden_states']
        teacher_neg_hidden = get_cor_teacher(teacher_neg_hidden,student_neg_hidden)
        student_neg_attn = student_attentions['itm_neg_attentions']
        teacher_neg_attn = teacher_attentions['itm_neg_attentions']
        teacher_neg_attn = get_cor_teacher(teacher_neg_attn,student_neg_attn,is_attn=True)
        itm_neg_hidden_loss = get_kd_loss(student_neg_hidden,teacher_neg_hidden,False,mse_loss,device)
        itm_neg_attn_loss = get_kd_loss(student_neg_attn,teacher_neg_attn,True,mse_loss,device)
        #mlm 
        student_mlm_hidden = student_hidden['mlm_hidden_states']
        teacher_mlm_hidden = teacher_hidden['mlm_hidden_states']
        teacher_mlm_hidden = get_cor_teacher(teacher_mlm_hidden,student_mlm_hidden)
        student_mlm_attn = student_attentions['mlm_attentions']
        teacher_mlm_attn = teacher_attentions['mlm_attentions']
        teacher_mlm_attn = get_cor_teacher(teacher_mlm_attn,student_mlm_attn,is_attn=True)
        mlm_hidden_loss = get_kd_loss(student_mlm_hidden,teacher_mlm_hidden,False,mse_loss,device)
        mlm_attn_loss = get_kd_loss(student_mlm_attn,teacher_mlm_attn,True,mse_loss,device)
        
        #logits kd loss
        #mlm head
        student_mlm_logits = student_logits['mlm_logits']
        teacher_mlm_logits= teacher_logits['mlm_logits']
        mlm_logits_loss = soft_cross_entropy(student_mlm_logits/args.temperature, teacher_mlm_logits/args.temperature)
        #itm head
        student_itm_logits = student_logits['itm_head_logits']
        teacher_itm_logits = teacher_logits['itm_head_logits']
        itm_logits_loss = soft_cross_entropy(student_itm_logits/args.temperature, teacher_itm_logits/args.temperature)


        loss = student_outputs['loss']
        loss_small = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm']
        loss_text_kd = text_attention_loss + text_hidden_loss
        loss_img_kd = image_attention_loss + 0.1 * image_hidden_loss
        loss_cross_kd = itm_neg_attn_loss + itm_neg_hidden_loss + itm_pos_attn_loss + itm_pos_hidden_loss + mlm_attn_loss + mlm_hidden_loss
        loss_kd =  itm_logits_loss + mlm_logits_loss + loss_text_kd + loss_img_kd + loss_cross_kd
        
        loss_in_total = loss_small*0.6 + loss_kd*0.4

        # print('#######loss_small : ',loss_small)
        # print('#######loss_kd : ',loss_kd)

        accelerator.backward_step(loss_in_total, optimizer)

        accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
        if accelerator_clip_grad_norm > 0:
            accelerator.optimizer_step(optimizer, student_model, accelerator_clip_grad_norm)
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itc=loss['loss_itc'].item())
        metric_logger.update(loss_itm=loss['loss_itm'].item())
        metric_logger.update(loss_itm_kd=itm_logits_loss.item())
        metric_logger.update(loss_mlm=loss['loss_mlm'].item())
        metric_logger.update(loss_mlm_kd=mlm_logits_loss.item())
        metric_logger.update(loss_text_kd=loss_text_kd.item())
        metric_logger.update(loss_img_kd=loss_img_kd.item())
        metric_logger.update(loss_cross_kd=loss_cross_kd.item())
        metric_logger.update(loss_kd=loss_kd.item())
        metric_logger.update(loss_small=loss_small.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_large=optimizer.param_groups[2]["lr"])

        if utils.is_main_process():
            current_epoch = global_step // step_per_epoch
            train_stats = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

            if (global_step+1) % step_per_epoch == 0:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch,
                             }

                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")


            if (global_step+1) % config['ckpt_frequent_step'] == 0:
                model_without_ddp = student_model
                if hasattr(student_model, 'module'):
                    model_without_ddp = student_model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'config': config,
                    'epoch': current_epoch,
                }

                checkpointer.save_checkpoint(model_state=save_obj,
                                            epoch=current_epoch, step=global_step,
                                            training_states=optimizer.state_dict())

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    config['train_file'] = ','.join(config['train_file'])
    config['train_file_regions'] = ','.join(config['train_file_regions'])
    config['batch_size'] = config['images']['batch_size']


    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset", flush=True)
    general_dataset, region_dataset = create_dataset('pretrain', config)

    if utils.is_main_process():
        print(f"### train_file: {config['train_file']}", flush=True)
        print(f"### train_file_regions: {config['train_file_regions']}", flush=True)
        print(f"### batch size, {config['batch_size']} x {int(os.environ.get('WORLD_SIZE', 1))}")

    general_loader = torch.utils.data.DataLoader(general_dataset, batch_size=config['images']['batch_size'],
                                               num_workers=config['images']['num_workers'],
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=general_dataset.collate_fn)

    region_loader = torch.utils.data.DataLoader(region_dataset, batch_size=config['regions']['max_images'],  # batch_size = max_images * max_regions
                                               num_workers=config['regions']['num_workers'],
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=region_dataset.collate_fn)

    print("Creating student model", flush=True)
    model = XVLM(config=config)
    
    model = model.to(device)
    print('Creating teacher model',flush=True)
    teacher_config = config
    teacher_config['vision_config'] = 'configs/config_clipvitB.json'
    teacher_config['text_num_hidden_layers'] = 12
    teacher_model = XVLM(config=teacher_config)
    if args.teacher_chkpt:
        teacher_model.load_pretrained(args.teacher_chkpt, teacher_config, is_eval=True)
    teacher_model.to(device)
    # print(model)
    print("### student model Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    print("### teacher model Total Params: ", sum(p.numel() for p in teacher_model.parameters() if p.requires_grad), flush=True)

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    # arg_sche['step_per_epoch'] = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
    arg_sche['step_per_epoch'] = math.ceil(100/(config['batch_size'] * world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = ApexDDPAccelerator(arg_acc, logger=None)
    start_epoch = 0

    #if hexists latest chkpt
    if args.resume:
        os.system('hdfs dfs -get {} ./data/'.format(os.path.join(args.output_dir,'training_state_latest.th')))       
        args.student_chkpt = './data/training_state_latest.th'
        checkpoint = torch.load(args.student_chkpt, map_location='cpu')                     
        print('resume training from latest checkpoint')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']+1
        model.load_pretrained(args.student_chkpt,config,is_eval=True)
            
    else:
        if args.student_chkpt:
            model.load_pretrained(args.student_chkpt,config,is_eval=True)

    model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)


    checkpointer = Checkpointer(args.output_dir)

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    max_epoch = config['schedular']['epochs']
    epoch_info = (start_epoch, max_epoch)

    print("Start training", flush=True)


    train(teacher_model, model, general_loader, region_loader, optimizer, epoch_info, device, lr_scheduler, config,
          accelerator, checkpointer)
    dist.barrier()

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)

    print('### Time {}'.format(total_time_str))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--student_chkpt',default=None,type=str)
    parser.add_argument('--teacher_chkpt',default=None,type=str)
    parser.add_argument('--temperature',default=1.0,type=float)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)


    if not hexists(args.output_dir):
        hmkdir(args.output_dir)
        print("args.output_dir: ", args.output_dir)
    
    if not hexists(os.path.join(args.output_dir,'training_state_latest.th')):
        args.resume = False
    
    if not hexists(os.path.join(args.output_dir,'config.yaml')):
        yaml.dump(config, open('config.yaml', 'w'))
        hcopy('config.yaml', args.output_dir)

    main(args, config)