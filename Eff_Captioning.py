import argparse
import os
import math
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_generation import XVLMForCaptioning
from efficient_models.model_generation import EffXVLMForCaptioning
import utils
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hexists
from torch.nn import MSELoss,KLDivLoss
from dataset.utils import collect_result, coco_caption_eval
from dataset import create_dataset, create_sampler, create_loader


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

def train(teacher_model,model, data_loader, optimizer_list,global_step, epoch, device, scheduler, config):
    model.train()
    with torch.no_grad():
        teacher_model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_samll', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_img_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_decoder_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_logits_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('lagrangian_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    optimizer,l0_optimizer,lagrangian_optimizer = optimizer_list

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)

        student_outputs = model(image, caption,output_attentions=True,output_hidden_states=True)

        with torch.no_grad():
            teacher_outputs = teacher_model(image, caption,output_attentions=True,output_hidden_states=True)
        
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
        
        #image kd
        student_image_hidden = student_hidden['image_hidden_states']
        teacher_image_hidden = teacher_hidden['image_hidden_states']
        teacher_image_hidden = get_cor_teacher(teacher_image_hidden,student_image_hidden)
        student_image_attn = student_attentions['image_attentions']
        teacher_image_attn = teacher_attentions['image_attentions']
        teacher_image_attn = get_cor_teacher(teacher_image_attn,student_image_attn,is_attn=True)
        image_hidden_loss = get_kd_loss(student_image_hidden,teacher_image_hidden,False,mse_loss,device,is_img=True)
        image_attention_loss = get_kd_loss(student_image_attn,teacher_image_attn,True,mse_loss,device)

        #decoder kd
        student_decoder_hidden = student_hidden['decoder_hidden_states']
        teacher_decoder_hidden = teacher_hidden['decoder_hidden_states']
        teacher_decoder_hidden = get_cor_teacher(teacher_decoder_hidden,student_decoder_hidden)
        student_decoder_attn = student_attentions['decoder_attentions']
        teacher_decoder_attn = teacher_attentions['decoder_attentions']
        teacher_decoder_attn = get_cor_teacher(teacher_decoder_attn,student_decoder_attn,is_attn=True)
        student_decoder_cross = student_cross_attentions['decoder_cross_attentions']
        teacher_decoder_cross = teacher_cross_attentions['decoder_cross_attentions']
        teacher_decoder_cross = get_cor_teacher(teacher_decoder_cross,student_decoder_cross,is_attn=True)
        decoder_hidden_loss = get_kd_loss(student_decoder_hidden,teacher_decoder_hidden,False,mse_loss,device,is_img=True)
        decoder_attention_loss = get_kd_loss(student_decoder_attn,teacher_decoder_attn,True,mse_loss,device)
        decoer_cross_loss = get_kd_loss(student_decoder_cross,teacher_decoder_cross,True,mse_loss,device)
        #logits kd
        student_logits = student_logits['logits']
        teacher_logits = teacher_logits['logits']
        logits_loss = soft_cross_entropy(student_logits/args.temperature, teacher_logits/args.temperature)
        
        loss_samll = student_outputs['loss']
        loss_img_kd = image_attention_loss + image_hidden_loss * 0.1
        loss_decoder_kd = decoder_attention_loss + decoder_hidden_loss + decoer_cross_loss
        loss_kd = logits_loss + loss_img_kd + loss_decoder_kd    #TODO:随便调调
        
        optimizer.zero_grad()
        loss = loss_kd*0.3 + loss_samll * 0.7
        #L0 regularisation
        lagrangian_loss = None
        lagrangian_loss, _, _ = model.module.l0_module.lagrangian_regularization(global_step)
        loss += lagrangian_loss
        
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
        
        metric_logger.update(loss_samll=loss_samll.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_img_kd=loss_img_kd.item())
        metric_logger.update(loss_decoder_kd=loss_decoder_kd.item())
        metric_logger.update(loss_logits_kd=logits_loss.item())
        metric_logger.update(lagrangian_loss=lagrangian_loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 50
    
    result = []

    for image, image_id in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device, non_blocking=True)

        captions = model_without_ddp.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if world_size > 8:
        assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size
        
    if args.lr != -1:
        config['schedular']['lr'] = args.lr
        config['optimizer']['lr'] = args.lr
    
    if args.reg_lr != -1:
        config['optimizer']['reg_learning_rate'] = args.reg_lr

    if args.sparsity is not None:
        config['sparsity'] = eval(args.sparsity) if isinstance(args.sparsity,str) else args.sparsity

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']

    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)
    datasets = [train_dataset, val_dataset, test_dataset]

    train_dataset_size = len(train_dataset)
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4, 4, 4], is_trains=[True, False, False],
                                              collate_fns=[None, None, None])

    print("Creating model")
    model = EffXVLMForCaptioning(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate, load_capt_pretrain=args.load_capt_pretrain)
    model = model.to(device)
    print('Creating teacher model',flush=True)
    teacher_config = config
    teacher_config['vision_config'] = 'configs/config_clipvitB.json'
    teacher_config['text_num_hidden_layers'] = 12
    teacher_config['num_dec_layers'] = 6
    teacher_model = XVLMForCaptioning(config=teacher_config)
    if args.teacher_chkpt:
        teacher_model.load_pretrained(args.teacher_chkpt, teacher_config,is_eval=True)
    teacher_model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("### Total Teacher Params: ", sum(p.numel() for p in teacher_model.parameters() if p.requires_grad))
    

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)

    if args.evaluate:
        print("Start evaluating")
        test_result = evaluation(model, test_loader, device, config)
        test_result_file = collect_result(test_result, 'test_eval', local_wdir=args.result_dir,
                                          hdfs_wdir=args.output_hdfs,
                                          write_to_hdfs=world_size > 8, save_result=True, remove_duplicate='image_id')

        if utils.is_main_process():
            coco_test = coco_caption_eval(config['test_gt_file'], test_result_file)
            log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
            print(log_stats, flush=True)

        dist.barrier()

    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)
        
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module 

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

        checkpointer = Checkpointer(args.output_hdfs if hexists(args.output_hdfs) else args.output_dir)
        global_step = 0
        for epoch in range(start_epoch, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(teacher_model,model, train_loader, optimizer_list, global_step,epoch, device, lr_scheduler, config)

            if utils.is_main_process():

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch}

                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    # 'epoch': epoch,
                }
                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=epoch,
                                             training_states=optimizer.state_dict())

            if epoch >= config['start_eval']:
                # val_result = evaluation(model, val_loader, device, config)
                # val_result_file = collect_result(val_result, 'val_epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                #                          write_to_hdfs=world_size > 8, save_result=True, remove_duplicate='image_id')

                test_result = evaluation(model, test_loader, device, config)
                test_result_file = collect_result(test_result, 'test_epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                         write_to_hdfs=world_size > 8, save_result=True, remove_duplicate='image_id')

                if utils.is_main_process():
                    # coco_val = coco_caption_eval(config['val_gt_file'], val_result_file)
                    coco_test = coco_caption_eval(config['test_gt_file'], test_result_file)

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 # **{f'val_{k}': v for k, v in coco_val.eval.items()},
                                 **{f'test_{k}': v for k, v in coco_test.eval.items()},
                                 'epoch': epoch}

                dist.barrier()

            if utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()

        os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--teacher_chkpt',default=None,type=str)
    parser.add_argument('--temperature',default=1.0,type=float)
    parser.add_argument('--load_capt_pretrain', action='store_true')
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--lr', default=-1,type=float)
    parser.add_argument('--reg_lr', default=-1,type=float)
    parser.add_argument('--sparsity', default=None,type=str)

    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    if hexists(args.output_dir):
        hmkdir(args.result_dir)
    else:
        hmkdir(args.output_dir)
        hmkdir(args.result_dir)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    main(args, config)