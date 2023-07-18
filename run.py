# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import sys
import time
import random
import argparse

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy


############ Set it correctly for distributed training across nodes
NNODES = 1  # e.g. 1/2/3/4
NPROC_PER_NODE = 1 # e.g. 8 gpus
MASTER_ADDR = 'SET_IT'
MASTER_PORT = 12345
NODE_RANK = 0  # e.g. 0/1/2
############

print("NNODES, ", NNODES)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NODE_RANK, ", NODE_RANK)

def get_nnodes(args):  # when using only part of nodes
    if args.dist == 'all':
        return NNODES

    elif args.dist == '2':
        assert NNODES >= 2
        return 2

    else:
        return 1


def get_dist_launch(args):  # some examples
    if args.dist == 'all':  # use all nodes
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    elif args.dist == '2':
        assert int(os.getenv("ARNOLD_WORKER_NUM")) >= 2
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes=2 --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    elif args.dist == '1':
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes=1 ".format(NPROC_PER_NODE)

    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.launch --nproc_per_node=1 " \
               "--nnodes=1 ".format(num)

    else:
        raise ValueError


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local

def run_pretrain(args, multilingual=False):
    print("### Start pre-training", flush=True)
    dist_launch = get_dist_launch(args)
    os.system(f"{dist_launch} --use_env {'Pretrain_multilingual.py' if multilingual else 'Pretrain.py'} --seed {args.seed} "
              f"--epoch {args.epoch} --config {args.config} --output_dir {args.output_dir}")

def run_pretrain_nlvr(args):
    print("### Start nlvr domain pre-training", flush=True)

    dist_launch = get_dist_launch(args)

    if len(args.load_ckpt_from):
        print(f"### Loading domain pre-trained results from: {args.load_ckpt_from}")
        domain_ckpt = get_from_hdfs(args.load_ckpt_from)

    else:  # domain pre-train
        if not os.path.exists(args.config): args.config = f'configs/{args.model}/NLVR_pretrain_O1.yaml'

        os.system(f"{dist_launch} --use_env NLVR_pretrain.py --seed {args.seed} --config {args.config} "
                  f"--output_dir {args.output_dir} --checkpoint {args.checkpoint}")

        domain_ckpt = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")

    return domain_ckpt


def run_pretrain_captioning(args):
    print("### Start captioning domain pre-training", flush=True)

    dist_launch = get_dist_launch(args)

    if len(args.load_ckpt_from):
        print(f"### Loading domain pre-trained results from: {args.load_ckpt_from}")
        domain_ckpt = get_from_hdfs(args.load_ckpt_from)

    else:  # domain pre-train
        if not os.path.exists(args.config): args.config = f'configs/{args.model}/Captioning_pretrain_O1.yaml'

        os.system(f"{dist_launch} --use_env Captioning_pretrain.py --seed {args.seed} --config {args.config} "
                  f"--output_dir {args.output_dir} --checkpoint {args.checkpoint}")

        domain_ckpt = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")

    return domain_ckpt


def run_nlvr2(args, load_nlvr_pretrain=False):
    dist_launch = get_dist_launch(args)
    assert os.path.exists("images/nlvr2")

    if not os.path.exists(args.config): args.config = f'./configs/{args.model}/NLVR.yaml'

    print("### Training NLVR2", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env Eff_NLVR.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"{f'--teacher_chkpt {args.teacher_chkpt}' if args.teacher_chkpt else ''} "
              f"--checkpoint {args.checkpoint} {'--load_nlvr_pretrain' if load_nlvr_pretrain else ''} "
              f"{'--evaluate' if args.evaluate else ''}")


def run_itr_coco(args):
    dist_launch = get_dist_launch(args)
    assert os.path.exists("images/coco")

    if not os.path.exists(args.config): args.config = f"configs/{args.model}/Retrieval_coco.yaml"

    print("### Training Retrieval COCO", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env 'Eff_Retrieval.py' --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_vqa(args):
    dist_launch = get_dist_launch(args)
    assert os.path.exists("images/coco") and os.path.exists("images/visualgenome")

    print("### Training VQA", flush=True)
    if not os.path.exists(args.config): args.config = f'./configs/{args.model}/VQA.yaml'

    os.system(f"{dist_launch} "
              f"--use_env Eff_VQA.py --config {args.config}"
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"{f'--teacher_chkpt {args.teacher_chkpt}' if args.teacher_chkpt else ''} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_coco_captioning(args, load_capt_pretrain=False):
    dist_launch = get_dist_launch(args)
    assert os.path.exists("images/coco")

    print("### Training COCO Captioning", flush=True)

    if not os.path.exists(args.config):
        args.config = f'./configs/{args.model}/Captioning.yaml'

    os.system(f"{dist_launch} "
              f"--use_env 'Eff_Captioning.py' --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --epoch {args.epoch} --checkpoint {args.checkpoint} "
              f"{f'--teacher_chkpt {args.teacher_chkpt}' if args.teacher_chkpt else ''} "
              f"{'--load_capt_pretrain' if load_capt_pretrain else ''} {'--evaluate' if args.evaluate else ''}")

def run_gd(args):
    print("### Start general distillation and pre-training", flush=True)
    dist_launch = get_dist_launch(args)
    os.system(f"{dist_launch} --use_env 'GeneralDistill.py' --seed {args.seed} "
            f"--epoch {args.epoch} --config {args.config} --output_dir {args.output_dir} "
            f"{f'--teacher_chkpt {args.teacher_chkpt}' if args.teacher_chkpt else ''} "
            f"{f'--student_chkpt {args.student_chkpt}' if args.student_chkpt else ''} "
            f"{f'--resume' if args.resume else ''} ") 


def run(args):
    if args.task == 'pretrain_4m_base':
        args.config = 'configs/Pretrain_XVLM_base_clipvit_4m.yaml'
        run_pretrain(args)

    elif args.task == 'pretrain_4m_small':
        args.config = 'configs/Pretrain_XVLM_small_4m.yaml'
        run_pretrain(args)
    
    elif args.task == 'gd_4m_small':
        args.config = './configs/Pretrain_XVLM_small_4m.yaml'
        run_gd(args)

    elif args.task == 'itr_coco':
        run_itr_coco(args)

    elif args.task == 'vqa_480':
        args.config = f"configs/{args.model}/VQA_480.yaml"
        run_vqa(args)

    elif args.task == 'nlvr_domain':
        domain_ckpt = run_pretrain_nlvr(args)

        # run fine-tune, reset args
        args.checkpoint = domain_ckpt
        if hexists(args.output_dir): args.output_dir = os.path.join(args.output_dir, 'nlvr_ft')
        args.config = f'./configs/{args.model}/NLVR.yaml'
        run_nlvr2(args, load_nlvr_pretrain=True)

    elif args.task == 'nlvr':
        run_nlvr2(args)

    elif args.task.startswith('coco_capt_domain'):
        domain_ckpt = run_pretrain_captioning(args)
        # run fine-tune, reset args
        args.checkpoint = domain_ckpt
        if hexists(args.output_dir): args.output_dir = os.path.join(args.output_dir, 'coco_capt_ft')
        args.config = f'./configs/{args.model}/Captioning.yaml'
        run_coco_captioning(args, load_capt_pretrain=True)

    elif args.task == 'coco_captioning':
        run_coco_captioning(args, load_capt_pretrain=False)

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")

    parser.add_argument('--config', default='', type=str, help="if not given, use default")
    parser.add_argument('--model', default='x-vlm-small-ft', type=str, help="to set default fine-tuning configs")

    parser.add_argument('--epoch', default=-1, type=int, help="for pre-training (debug) only")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                           "this option only works for fine-tuning scripts.")

    parser.add_argument('--checkpoint', default='', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default='', type=str, help="load domain pre-trained params")

    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, required=True, help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--output_hdfs', type=str, default='', help="HDFS path required by VQA and Refcoco, "
                                                                    "to collect eval results among nodes")

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--teacher_chkpt', default=None, type=str)
    parser.add_argument('--student_chkpt', default=None, type=str)
    parser.add_argument('--resume', action='store_true', help="for pretraining, to resume from latest checkpoint")

    args = parser.parse_args()

    if MASTER_ADDR == 'SET_IT':
        print("### warning: the settings for distributed training is not filled (ignore this if you only use one node)")

    if '/SET/PATH/TO/hadoop/bin/hdfs' in HADOOP_BIN:
        print("### warning: you have not set the path to hadoop_bin (ignore this if you don't use HDFS)")

    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)

    if len(args.output_hdfs):
        assert hexists(os.path.dirname(args.output_hdfs))

    if len(args.config):
        assert hexists(args.config)

        if args.config.startswith('hdfs://'):
            args.config = get_from_hdfs(args.config)

    if args.checkpoint.startswith('hdfs://'):
        args.checkpoint = get_from_hdfs(args.checkpoint)

    run(args)

