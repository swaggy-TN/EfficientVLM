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

from efficient_models.model_generation import EffXVLMForVQA
from models.model_generation import XVLMForVQA
import utils
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hexists,hcopy,hopen

from dataset.utils import collect_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn, build_tokenizer
from torch.nn import MSELoss,KLDivLoss
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


def train(teacher_model,model,data_loader,  optimizer_list, tokenizer, global_step, epoch,  device, scheduler, config,stop_prune=False):
    model.train()
    with torch.no_grad():
        teacher_model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('global_step', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_samll', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_text_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_img_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cross_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_decoder_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_logits_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lagrangian', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    optimizer,l0_optimizer,lagrangian_optimizer = optimizer_list


    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        
        student_outputs = model(image, question_input, answer_input, train=True, k=n, weights=weights,output_attentions=True,output_hidden_states=True,stop_prune=stop_prune)
        with torch.no_grad():
            teacher_outputs = teacher_model(image, question_input, answer_input, train=True, k=n, weights=weights,output_attentions=True,output_hidden_states=True)
        
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
        loss_text_kd =  text_attention_loss + text_hidden_loss
        loss_img_kd = image_attention_loss + image_hidden_loss * 0.2
        loss_cross_kd = (cross_hidden_loss + cross_self_attention_loss + cross_attention_loss) *0.5
        loss_decoder_kd = decoder_attention_loss + decoder_hidden_loss + decoer_cross_loss
        loss_kd = logits_loss + loss_text_kd + loss_img_kd +  loss_cross_kd + loss_decoder_kd
        loss = loss_kd*0.4 + loss_samll * 0.6
         #L0 regularisation
        lagrangian_loss = None

        lagrangian_loss, _, _ = model.module.l0_module.lagrangian_regularization(global_step)
        loss += lagrangian_loss
        
        # module部分的zero_grad
        model.zero_grad()

        model.module.l0_module.zero_grad()  
    #optimizer部分的zero_grad
        l0_optimizer.zero_grad()
        lagrangian_optimizer.zero_grad()

        optimizer.zero_grad()
        #整体loss的backward
        loss.backward()
        #更新梯度
        optimizer.step()
    
        l0_optimizer.step()
        lagrangian_optimizer.step()
        scheduler.step()
    
        model.module.l0_module.constrain_parameters()        

        global_step += 1

        metric_logger.update(global_step=global_step)
        metric_logger.update(loss_samll=loss_samll.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_img_kd=loss_img_kd.item())
        metric_logger.update(loss_text_kd=loss_text_kd.item())
        metric_logger.update(loss_cross_kd=loss_cross_kd.item())
        metric_logger.update(loss_decoder_kd=loss_decoder_kd.item())
        metric_logger.update(loss_logits_kd=logits_loss.item())
        metric_logger.update(loss_lagrangian=lagrangian_loss.item())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    answer_input = tokenizer(data_loader.dataset.answer_list, padding='longest', return_tensors='pt').to(device)

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        
        with torch.no_grad():
            topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])      

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})   

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if world_size > 8:
        assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size
    
    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

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
    if args.epoch > -1:
        max_epoch = args.epoch
    max_epoch = config['schedular']['epochs']

    print("Creating vqa datasets")
    train_dataset, vqa_test_dataset = create_dataset('vqa', config)
    datasets = [train_dataset, vqa_test_dataset]

    train_dataset_size = len(train_dataset)
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    print("Creating model")
    tokenizer = build_tokenizer(config['text_encoder'])

    print("### pad_token_id, ", train_dataset.pad_token_id)
    print("### eos_token, ", train_dataset.eos_token)
    config['pad_token_id'] = train_dataset.pad_token_id
    config['eos'] = train_dataset.eos_token
    model = EffXVLMForVQA(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate or args.load_vqa_pretrain)
    model = model.to(device)
    print('Creating teacher model',flush=True)
    teacher_config = config
    teacher_config['vision_config'] = 'configs/config_clipvitB.json'
    teacher_config['text_num_hidden_layers'] = 12
    teacher_config['num_dec_layers'] = 6
    teacher_model = XVLMForVQA(config=teacher_config)
    if args.teacher_chkpt:
        teacher_model.load_pretrained(args.teacher_chkpt, teacher_config,is_eval=True)
    teacher_model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("### Total Teacher Params: ", sum(p.numel() for p in teacher_model.parameters() if p.requires_grad))
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)

    if args.evaluate:
        print("Start evaluating")
        vqa_result = evaluation(model, test_loader, tokenizer, device, config)
        
        result = collect_result(vqa_result, 'vqa_eval', local_wdir=args.result_dir,
                                hdfs_wdir=args.output_hdfs,
                                write_to_hdfs=world_size > 8, save_result=True)

        dist.barrier()

    else:
        print("Start training", flush=True)
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        l0_optimizer, lagrangian_optimizer = create_L0_optimizer(arg_opt,model_without_ddp.l0_module)
        optimizer_list = (optimizer,l0_optimizer,lagrangian_optimizer)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(config['batch_size_train']*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

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
            if epoch >= args.stop_prune_epoch:
                train_stats = train(teacher_model, model, train_loader, optimizer_list, tokenizer, global_step, epoch, device, lr_scheduler, config,stop_prune=True)
            else:
                train_stats = train(teacher_model, model, train_loader, optimizer_list, tokenizer, global_step, epoch, device, lr_scheduler, config,stop_prune=False)

            if utils.is_main_process():
                # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                #              'epoch': epoch,
                #             }
                # with hopen(os.path.join(args.output_dir, "log.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")

                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module
                
                with torch.no_grad():
                    zs = model_without_ddp.l0_module.forward(training=False)
                    pruned_model_size_info = model_without_ddp.l0_module.calculate_model_size(zs)
                pruned_model_size_info['pruned_model_sparsity']

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=epoch,
                                             training_states=optimizer.state_dict())

            if epoch >= config['start_eval']:
                vqa_result = evaluation(model, test_loader, tokenizer, device, config)
                result = collect_result(vqa_result, 'vqa_result_epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                         write_to_hdfs=world_size >= 8, save_result=True)

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
    parser.add_argument('--teacher_chkpt',default=None,type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--temperature',default=1.0,type=float)
    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--stop_prune_epoch', default=10, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--load_vqa_pretrain', action='store_true')
    parser.add_argument('--lr', default=-1,type=float)
    parser.add_argument('--reg_lr', default=-1,type=float)
    parser.add_argument('--sparsity', default=None,type=str)

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')
    hmkdir(args.output_dir)
    hmkdir(args.result_dir)

    if args.output_hdfs and not hexists(args.output_hdfs):
        hmkdir(args.output_hdfs)
        print("args.output_hdfs: ", args.output_hdfs)

    if not hexists(args.output_dir):
        hmkdir(args.output_dir)
        print("args.output_dir: ", args.output_dir)
    
    if not hexists(os.path.join(args.output_hdfs,'config.yaml')):
        yaml.dump(config, open('config.yaml', 'w'))
        hcopy('config.yaml', args.output_dir)

    main(args, config)
