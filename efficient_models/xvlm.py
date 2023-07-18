# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.
import os
from pickle import NONE
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from functools import partial

from models import box_ops

from models.vit import interpolate_pos_embed
from efficient_models.eff_bert import BertConfig,BertForMaskedLM,BertModel
from efficient_models.eff_vit import CLIPVisionTransformer

from utils import read_json


def load_params_change_prefix(state_dict: dict, prefix: str, new_prefix: str):
    if prefix == new_prefix:
        return state_dict

    state_dict_new = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k.replace(prefix, new_prefix)

        state_dict_new[k] = v

    return state_dict_new


def load_params_choose_layers(prefix: str, state_dict: dict, mapper: dict):
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            new_k = None
            for i in mapper.keys():
                if k.startswith(f'{prefix}.{i}.'):
                    new_k = k.replace(f'{prefix}.{i}.', f'{prefix}.{mapper[i]}.')
                    break
            if new_k:
                state_dict[new_k] = state_dict[k]

            del state_dict[k]

    return state_dict


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


def build_vision_encoder(config, load_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2

    if config['use_clip_vit']:  # good performance, but only base model available
        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        vision_encoder = CLIPVisionTransformer(image_size=config['image_res'], patch_size=vision_config['patch_size'],
                                               hidden_size=vision_config['vision_width'],
                                               hidden_act=vision_config['hidden_act'],
                                               num_attention_heads=vision_config['num_attention_heads'],
                                               attention_dropout=vision_config['attention_dropout'],
                                               intermediate_size=vision_config['intermediate_size'],
                                               num_hidden_layers=vision_config['num_hidden_layers'],
                                               local_attn_depth=vision_config['local_attn_depth'])

        if load_params:
            # download from https://huggingface.co/openai/clip-vit-base-patch16/tree/main
            state_dict_orig = torch.load(vision_config['ckpt'], map_location="cpu")
            state_dict = {}
            for k, v in state_dict_orig.items():
                if k.startswith('vision_model.'):
                    k = k[13:]
                    if k.startswith('embeddings.'):
                        k = k[11:]
                        k = k.replace('patch_embedding.weight', 'patch_embed.weight')
                        k = k.replace('position_embedding.weight', 'pos_embed.weight')

                    if k != 'position_ids':
                        state_dict[k] = v

            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed.weight'].unsqueeze(dim=0),
                                                       num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

            assert vision_config['num_hidden_layers'] in [6, 12], "param initialization not implemented"
            if vision_config['num_hidden_layers'] == 6:
                mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                load_params_choose_layers('encoder.layers', state_dict, mapper)
    else:
        raise NotImplementedError

    if load_params:
        print("### Load ViT: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

    return vision_encoder, vision_width


def build_text_encoder(config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
    init_params = []  # train from scratch with larger lr

    if config_text is None:
        config_text = BertConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))
        config_text.num_hidden_layers = config['text_num_hidden_layers'] if 'text_num_hidden_layers' in config else 12
        assert config_text.num_hidden_layers in [6, 12], "param initialization not implemented"
        config_text.fusion_layer = config_text.num_hidden_layers // 2
    else:
        assert isinstance(config_text, BertConfig)

    # set configs
    
    config_text.encoder_width = vision_width

    if use_mlm_loss:  # for pre-training, load_text_params by default (otherwise notimplemented)
        assert load_text_params is True
        if ('accelerator' in config.keys()) and (config['accelerator']['FP16_OPT_LEVEL'] != 'O0'):
            config_text.fp16 = True  # will use some operations to avoid gradient overflow

        text_encoder = BertForMaskedLM(config=config_text)

        print("### Initializing text encoder from ", os.path.join(config['text_encoder'], 'pytorch_model.bin'))
        state_dict = torch.load(os.path.join(config['text_encoder'], 'pytorch_model.bin'))

        if config_text.num_hidden_layers == 6:
            mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
            load_params_choose_layers('bert.encoder.layer', state_dict, mapper)

        msg = text_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys, flush=True)
        print("unexpected_keys: ", msg.unexpected_keys, flush=True)
        init_params += [f'text_encoder.{k}' for k in msg.missing_keys]

    else:  # for fine-tuning, not load_text_params by default
        assert load_text_params is False
        text_encoder = BertModel(config=config_text, add_pooling_layer=False)

    return text_encoder, init_params


def load_pretrained(ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

    if is_eval:
        return state_dict

    num_patches = (config['image_res'] // config['patch_size']) ** 2

    print("### Loading pretrained vision encoder", flush=True)
    if config['use_clip_vit']:
        del state_dict['vision_encoder.position_ids']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed.weight'].unsqueeze(dim=0),
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if key.startswith('text_encoder.'):
                if 'bert.' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

    return state_dict


class XVLMBase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=False,
                 use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                 config_text=None):
        super().__init__()
        self.init_params = []  # train from scratch with larger lr

        self.vision_encoder, vision_width = build_vision_encoder(config, load_params=load_vision_params)

        self.text_encoder, init_params = build_text_encoder(config, vision_width=vision_width, load_text_params=load_text_params,
                                                            use_mlm_loss=use_mlm_loss,
                                                            config_text=config_text)  # text & cross-modal
        self.init_params.extend(init_params)
        self.num_text_layers = self.text_encoder.config.fusion_layer
        self.num_cross_layers = self.text_encoder.config.num_hidden_layers - self.num_text_layers

        self.vision_width = vision_width
        self.text_width = self.text_encoder.config.hidden_size  # i.e. cross_width

        if use_contrastive_loss:
            self.embed_dim = config['embed_dim']
            self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)
            self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
            self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])

            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            self.init_params.extend(['temp'])

        if use_matching_loss:
            self.itm_head = build_mlp(input_dim=self.text_width, output_dim=2)
            self.init_params.extend(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])

        if use_bbox_loss:
            self.bbox_head = build_mlp(input_dim=self.text_width, output_dim=4)
            self.init_params.extend(['bbox_head.' + n for n, _ in self.bbox_head.named_parameters()])

        # check
        named_parameters = set([n for n, _ in self.named_parameters()])
        for n in set(self.init_params):
            if n not in named_parameters:
                print(f"warning: {n} not in named_parameters")
                self.init_params.remove(n)

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def get_vision_embeds(self, image, image_atts=None, idx_to_group_img=None,output_attentions=None,output_hidden_states=None,head_z=None,head_layer_z=None,mlp_z=None):
        """
        vision_embeds: cls + patch embeds
        """
        if idx_to_group_img is None:
            if not output_attentions:
                image_embeds = self.vision_encoder(image,head_z=head_z,head_layer_z=head_layer_z,mlp_z=mlp_z)[0]
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                return image_embeds, image_atts
            else:
                image_embeds,image_hidden_states,image_all_attentions = self.vision_encoder(image,output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                                                                                head_z=head_z,head_layer_z=head_layer_z,mlp_z=mlp_z)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                return image_embeds, image_atts,image_hidden_states,image_all_attentions  # full attention

        else:  # image < bsz
            if image_atts is None:
                image_embeds_fullatts = self.vision_encoder(image,head_z=head_z,head_layer_z=head_layer_z,mlp_z=mlp_z)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))  # expend to bsz

                image_atts = torch.ones(image_embeds_fullatts.size()[:-1], dtype=torch.long).to(image.device)

                return image_embeds_fullatts, image_atts

            else:
                assert image_atts.size(0) == idx_to_group_img.size(0)  # bsz
                image_embeds,image_hidden_states,image_all_attentions,image_embeds_fullatts = \
                    self.vision_encoder(image, idx_to_group_img=idx_to_group_img, image_atts=image_atts,
                    output_attentions=output_attentions,output_hidden_states=output_hidden_states,head_z=head_z,head_layer_z=head_layer_z)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))

                return image_embeds, image_atts, image_embeds_fullatts,image_hidden_states,image_all_attentions

    def get_text_embeds(self, text_ids, text_atts,output_attentions=None,output_hidden_states=None,head_z=None,head_layer_z=None,mlp_z=None):
        assert output_hidden_states == output_attentions

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder

        outputs = encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text',
                            output_attentions=output_attentions,output_hidden_states=output_hidden_states,head_z=head_z,head_layer_z=head_layer_z,mlp_z=mlp_z)
        if output_attentions:
            return outputs.last_hidden_state, outputs.hidden_states, outputs.attentions
        
        return outputs.last_hidden_state

    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None,output_hidden_states=None,output_attentions=None,head_z=None,head_layer_z=None,mlp_z=None):
        assert text_atts is not None
        assert output_attentions == output_hidden_states

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder

        if text_embeds is not None:
            if not output_attentions:
                return encoder(encoder_embeds=text_embeds,
                            attention_mask=text_atts,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=True,
                            mode='fusion',
                            head_z=head_z,
                            head_layer_z=head_layer_z,
                            mlp_z=mlp_z
                            ).last_hidden_state
            else:
                outputs = encoder(encoder_embeds=text_embeds,
                            attention_mask=text_atts,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=True,
                            mode='fusion',
                            head_z=head_z,
                            head_layer_z=head_layer_z,
                            mlp_z=mlp_z)
                return outputs.last_hidden_state,outputs.hidden_states,outputs.attentions,outputs.cross_attentions
        
        elif text_ids is not None:
            if not output_attentions:
                return encoder(text_ids,
                            attention_mask=text_atts,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                            head_z=head_z,
                            head_layer_z=head_layer_z,
                            mlp_z=mlp_z
                            ).last_hidden_state
            else:
                outputs = encoder(text_ids,
                            attention_mask=text_atts,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            head_z=head_z,
                            head_layer_z=head_layer_z,
                            mlp_z=mlp_z)
                return outputs.last_hidden_state,outputs.hidden_states,outputs.attentions,outputs.cross_attentions
        else:
            raise ValueError

    def get_features(self, image_embeds=None, text_embeds=None):
        if image_embeds is None:
            return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        else:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1), \
                   F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None,output_attentions=None,output_hidden_states=None,head_z=None,head_layer_z=None,mlp_z=None):
        """
        Matching Loss with hard negatives
        """
        bs = image_embeds.size(0)
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        if output_hidden_states:
            pos_last_hidden,pos_hidden_states,pos_attentions,pos_cross_attentions = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, text_atts=text_atts,
                                            output_attentions=output_attentions,output_hidden_states=output_hidden_states,head_z=head_z,head_layer_z=head_layer_z,mlp_z=mlp_z)
            cross_pos = pos_last_hidden[:,0,:]
            neg_last_hidden,neg_hidden_states,neg_attentions,neg_cross_attentions = self.get_cross_embeds(image_embeds_all, image_atts_all, text_embeds=text_embeds_all,
                                          text_atts=text_atts_all,output_attentions=output_attentions,output_hidden_states=output_hidden_states,head_z=head_z,head_layer_z=head_layer_z,mlp_z=mlp_z)
            cross_neg = neg_last_hidden[:,0,:]
        else:
            cross_pos = self.get_cross_embeds(image_embeds, image_atts,text_embeds=text_embeds, text_atts=text_atts,
                                            head_z=head_z,head_layer_z=head_layer_z,mlp_z=mlp_z)[:, 0,:]
            cross_neg= self.get_cross_embeds(image_embeds_all, image_atts_all, text_embeds=text_embeds_all,text_atts=text_atts_all,
                                            head_z=head_z,head_layer_z=head_layer_z,mlp_z=mlp_z)[:, 0, :]


        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)

        matching_loss = F.cross_entropy(output, itm_labels)
        if not output_hidden_states:
            return matching_loss
        else:
            res = {'loss':matching_loss,'pos_hidden_states':pos_hidden_states,'neg_hidden_states':neg_hidden_states,
                    'pos_attentions':pos_attentions,'neg_attentions':neg_attentions,'pos_cross_attentions':pos_cross_attentions,
                    'neg_cross_attentions':neg_cross_attentions,'logits':output}
            return res

    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids,output_attentions=None,output_hidden_states=None,head_z=None,head_layer_z=None,mlp_z=None):
        assert output_hidden_states == output_attentions
        if not output_attentions:
            return self.text_encoder(text_ids_masked,
                                    attention_mask=text_atts,
                                    encoder_hidden_states=image_embeds,
                                    encoder_attention_mask=image_atts,
                                    return_dict=True,
                                    labels=masked_ids,
                                    masked_pos=masked_pos,
                                    head_z=head_z,
                                    head_layer_z=head_layer_z,
                                    mlp_z=mlp_z).loss
        else:
            outputs = self.text_encoder(text_ids_masked,
                                    attention_mask=text_atts,
                                    encoder_hidden_states=image_embeds,
                                    encoder_attention_mask=image_atts,
                                    return_dict=True,
                                    labels=masked_ids,
                                    masked_pos=masked_pos,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    head_z=head_z,
                                    head_layer_z=head_layer_z,
                                    mlp_z=mlp_z)
            return outputs.loss,outputs.logits,outputs.hidden_states,outputs.attentions,outputs.cross_attentions

    def predict_bbox(self, image_embeds, text_embeds, text_atts,output_attentions=None,output_hidden_states=None,head_z=None,head_layer_z=None,mlp_z=None):
        """
        Args:
            image_embeds: encoding full images

        Returns:
            output_coord: bsz, 4
        """
        assert image_embeds.size(0) == text_embeds.size(0)

        if output_attentions is not None:
            outputs = self.get_cross_embeds(image_embeds, torch.ones(image_embeds.shape[:2]).to(image_embeds.device),
                                           text_embeds=text_embeds, text_atts=text_atts,
                                           output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                           head_z=head_z,mlp_z=mlp_z)
            output_cls = outputs[0][:,0,:]
        else:
            output_cls = self.get_cross_embeds(image_embeds, torch.ones(image_embeds.shape[:2]).to(image_embeds.device),
                                                text_embeds=text_embeds, text_atts=text_atts,head_z=head_z,mlp_z=mlp_z)[:,0,:]
        output_coord = self.bbox_head(output_cls).sigmoid()

        return (output_coord,) + outputs[1:] if output_attentions else (output_coord,)

#TODO:bbox_loss尚未做logits和hiddenstates的实现
    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes
