import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from efficient_models.model_nlvr import EffXVLMForNLVR
from models.model_nlvr import XVLMForNLVR
from torch.nn import MSELoss,KLDivLoss
import utils
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer,create_L0_optimizer

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

def train(teacher_model,model, data_loader, optimizer_list, tokenizer,global_step,epoch, device, scheduler):
    model.train()
    with torch.no_grad():
        teacher_model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_samll', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_img_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_text_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cross_kd', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('lagrangian_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    optimizer,l0_optimizer,lagrangian_optimizer = optimizer_list

    for i, (image0, image1, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  

        student_outputs = model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=True,output_attentions=True,output_hidden_states=True)
        with torch.no_grad():
            teacher_outputs = teacher_model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=True,output_attentions=True,output_hidden_states=True)

        student_hidden = student_outputs['hidden_dict']
        teacher_hidden = teacher_outputs['hidden_dict']
        student_attentions = student_outputs['attention_dict']
        teacher_attentions = teacher_outputs['attention_dict']
        #just for cross kd
        student_cross_attentions = student_outputs['cross_attention_dict']
        teacher_cross_attentions = teacher_outputs['cross_attention_dict']
        student_logits = student_outputs['logits_dict']
        teacher_logits = teacher_outputs['logits_dict']
        mse_loss = MSELoss()

        #text kd
        student_text_hidden = student_hidden['text_hidden_states']
        teacher_text_hidden = teacher_hidden['text_hidden_states']
        teacher_text_hidden = get_cor_teacher(teacher_text_hidden,student_text_hidden)
        #后6层属于cross_encoder
        student_cross_hidden = student_text_hidden[4:]
        teacher_cross_hidden = teacher_text_hidden[4:]
        student_text_attn = student_attentions['text_attentions']
        teacher_text_attn = teacher_attentions['text_attentions']
        teacher_text_attn = get_cor_teacher(teacher_text_attn,student_text_attn,is_attn=True)
        student_cross_selfattention = student_text_attn[3:]
        teacher_cross_selfattention = teacher_text_attn[3:]
        student_cross_attn = student_cross_attentions['cross_attentions']
        teacher_cross_attn = teacher_cross_attentions['cross_attentions']
        teacher_cross_attn = get_cor_teacher(teacher_cross_attn,student_cross_attn,is_attn=True)
        text_hidden_loss = get_kd_loss(student_text_hidden[:4],teacher_text_hidden[:4],False,mse_loss,device)
        text_attention_loss = get_kd_loss(student_text_attn[:3],teacher_text_attn[:3],True,mse_loss,device)
        cross_hidden_loss = get_kd_loss(student_cross_hidden,teacher_cross_hidden,False,mse_loss,device)
        cross_self_attention_loss = get_kd_loss(student_cross_selfattention,teacher_cross_selfattention,True,mse_loss,device)
        cross_attention_loss = get_kd_loss(student_cross_attn,teacher_cross_attn,True,mse_loss,device)

        #image kd
        student_image_hidden = student_hidden['image_hidden_states']
        teacher_image_hidden = teacher_hidden['image_hidden_states']
        teacher_image_hidden = get_cor_teacher(teacher_image_hidden,student_image_hidden)
        student_image_attn = student_attentions['image_attentions']
        teacher_image_attn = teacher_attentions['image_attentions']
        teacher_image_attn = get_cor_teacher(teacher_image_attn,student_image_attn,is_attn=True)
        image_hidden_loss = get_kd_loss(student_image_hidden,teacher_image_hidden,False,mse_loss,device,is_img=True)
        image_attention_loss = get_kd_loss(student_image_attn,teacher_image_attn,True,mse_loss,device)

        student_logits = student_logits['cls_head_logits']
        teacher_logits = teacher_logits['cls_head_logits']
        logits_loss = soft_cross_entropy(student_logits/args.temperature, teacher_logits/args.temperature)

        loss_samll = student_outputs['loss']
        loss_text_kd =  text_attention_loss + text_hidden_loss
        loss_img_kd = image_attention_loss + image_hidden_loss * 0.1
        loss_cross_kd = (cross_hidden_loss + cross_self_attention_loss + cross_attention_loss) *0.5
        loss_kd = logits_loss + loss_text_kd + (loss_img_kd +  loss_cross_kd) * 0.33 #TODO:随便调调
        optimizer.zero_grad()
        loss = 0.8 * loss_samll + 0.2 * loss_kd
        #L0 regularisation
        lagrangian_loss = None
        lagrangian_loss, _, _ = model.module.l0_module.lagrangian_regularization(global_step)

        loss += lagrangian_loss
        
        #整体loss的backward
        loss.backward()
        #更新梯度
        optimizer.step()
        l0_optimizer.step()
        lagrangian_optimizer.step()
        
        scheduler.step()

        model.module.l0_module.constrain_parameters()

        # module部分的zero_grad
        model.zero_grad()
        model.module.l0_module.zero_grad()
        #optimizer部分的zero_grad
        optimizer.zero_grad()
        l0_optimizer.zero_grad()
        lagrangian_optimizer.zero_grad()
        
        global_step += 1

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_samll=loss_samll.item())
        metric_logger.update(loss_img_kd=loss_img_kd.item())
        metric_logger.update(loss_text_kd=loss_text_kd.item())
        metric_logger.update(loss_cross_kd=loss_cross_kd.item())
        metric_logger.update(lagrangian_loss=lagrangian_loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50


    for image0, image1, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        prediction = model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=False)
 
        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size'] = args.bs // world_size
    
    if args.lr != -1:
        config['schedular']['lr'] = args.lr
        config['optimizer']['lr'] = args.lr
    
    if args.reg_lr != -1:
        config['optimizer']['reg_learning_rate'] = args.reg_lr

    if args.sparsity is not None:
        config['sparsity'] = eval(args.sparsity)


    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('nlvr', config)
    datasets = [train_dataset, val_dataset, test_dataset]

    train_dataset_size = len(train_dataset)
    train_batch_size = config['batch_size']
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")
        print(f"### test data {len(test_dataset)}", flush=True)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']] * 3,
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])


    print("Creating model")
    model = EffXVLMForNLVR(config=config)
    model.load_pretrained(args.checkpoint, config, load_nlvr_pretrain=args.load_nlvr_pretrain, is_eval=args.evaluate)
    model = model.to(device)
    print('Creating teacher model',flush=True)
    teacher_config = config
    teacher_config['vision_config'] = 'configs/config_clipvitB.json'
    teacher_config['text_num_hidden_layers'] = 12
    teacher_model = XVLMForNLVR(config=teacher_config)
    if args.teacher_chkpt:
        teacher_model.load_pretrained(args.teacher_chkpt, teacher_config,is_eval=True,load_nlvr_pretrain=True)
    teacher_model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("### Total Teacher Params: ", sum(p.numel() for p in teacher_model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    tokenizer = build_tokenizer(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        val_stats = evaluate(model, val_loader, tokenizer, device)
        test_stats = evaluate(model, test_loader, tokenizer, device)

        if utils.is_main_process():
            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            print(log_stats)

        dist.barrier()

    else:
        print("Start training", flush=True)
        arg_opt = utils.AttrDict(config['optimizer'])        
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(train_batch_size*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        l0_optimizer, lagrangian_optimizer = create_L0_optimizer(arg_opt,model_without_ddp.l0_module)
        optimizer_list = (optimizer,l0_optimizer,lagrangian_optimizer)

        arg_l0 = utils.AttrDict(config['L0_schedular'])
        if arg_l0['lagrangian_warmup_epochs'] < 1:
            lagrangian_warmup_steps = int(arg_l0['lagrangian_warmup_epochs'] * arg_l0['epochs'] * arg_sche['step_per_epoch'])
        else:
            lagrangian_warmup_steps = arg_l0['lagrangian_warmup_epochs'] * arg_sche['step_per_epoch']
       
        model_without_ddp.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)


        # prepruning_finetune_steps = arg_l0['prepruning_finetune_steps'] * arg_sche['step_per_epoch']
        # print(f"Prepruning finetune steps: {prepruning_finetune_steps}")
        print(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0
        global_step = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(teacher_model,model, train_loader, optimizer_list, tokenizer, global_step,epoch, device, lr_scheduler)
            val_stats = evaluate(model, val_loader, tokenizer, device)
            test_stats = evaluate(model, test_loader, tokenizer, device)
            with torch.no_grad():
                zs = model_without_ddp.l0_module.forward(training=False)
            pruned_model_size_info = model_without_ddp.l0_module.calculate_model_size(zs)


            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'cur_sparsity':pruned_model_size_info['pruned_model_sparsity']
                            }

                if float(val_stats['acc']) > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = float(val_stats['acc'])
                    best_epoch = epoch

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/NLVR.yaml')
    parser.add_argument('--output_dir', default='output/nlvr')
    parser.add_argument('--teacher_chkpt',default=None,type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--temperature',default=1.0,type=float)
    parser.add_argument('--load_nlvr_pretrain', action='store_true')
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--lr', default=-1,type=float)
    parser.add_argument('--reg_lr', default=-1,type=float)
    parser.add_argument('--sparsity', default=None,type=str)
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)