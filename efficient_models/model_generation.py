import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

from dataset import build_tokenizer
from efficient_models.eff_bert import BertConfig, BertLMHeadModel
from efficient_models.xvlm import XVLMBase, load_pretrained
from efficient_models.generation_l0_module import VQAL0Module
from efficient_models.xvlm_l0_module import XVLML0Module
from utils import read_json
import time

def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

class EffXVLMForVQA(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=None)

        assert isinstance(config['pad_token_id'], int)
        self.pad_token_id = config['pad_token_id']
        config_enc = self.text_encoder.config

        self.num_text_layers = config_enc.fusion_layer
        self.num_cross_layers = config_enc.num_hidden_layers - config_enc.fusion_layer
        assert config['num_dec_layers'] == self.num_cross_layers, "initialization not implemented"

        config_dec = copy.deepcopy(config_enc)
        config_dec.encoder_width = config_enc.hidden_size
        config_dec.fusion_layer = 0  # start index
        config_dec.num_hidden_layers = config['num_dec_layers']
        self.cross_encoder_width = config_enc.encoder_width  # i.e. vision_width
        self.dec_encoder_width = config_enc.hidden_size
        self.text_decoder = BertLMHeadModel(config=config_dec)
        #L0module initialization
        self.l0_module = VQAL0Module(config,target_sparsity=config['sparsity'])

        if self.dec_encoder_width != self.cross_encoder_width:
            self.init_params = ['text_decoder.' + n for n, _ in self.text_decoder.named_parameters()
                                if ('crossattention.self.key' in n) or ('crossattention.self.value' in n)]
        else:
            self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(ckpt_rpath, config, load_text=False)

            print("### Loading pretrained text encoder", flush=True)
            for key in list(state_dict.keys()):
                if 'bert.' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]

                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                if 'text_encoder.' in key:
                    if 'layer.' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < self.num_text_layers:
                            del state_dict[key]
                            continue

                        elif (self.dec_encoder_width != self.cross_encoder_width) and \
                                (('crossattention.self.key' in key) or ('crossattention.self.value' in key)):
                            del state_dict[key]
                            continue

                        else:
                            decoder_layer_num = (layer_num - self.num_text_layers)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key

                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    state_dict[decoder_key] = state_dict[key]
                    del state_dict[key]


        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, quesiton, answer=None, k=None, weights=None, train=True,output_attentions=None,output_hidden_states=None,stop_prune=False):
        if train: #train一定output attentions
            if not stop_prune:        
                zs = self.l0_module.forward(training=True)
            else:
                with torch.no_grad():
                    zs = self.l0_module.forward(training=False)
            vision_head_z = zs['vision_head_z']
            vision_mlp_z = zs['vision_intermediate_z']
            text_head_z = zs['text_head_z']
            text_mlp_z = zs['text_intermediate_z']
            cross_head_z = zs['cross_head_z']
            cross_mlp_z = zs['cross_intermediate_z']
            decoder_head_z = zs['decoder_head_z']
            decoder_mlp_z = zs['decoder_intermediate_z']
            if output_attentions:
                image_embeds,image_hidden_states,image_attentions = self.vision_encoder(image,
                                                                                    output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                                                                    head_z=vision_head_z,mlp_z=vision_mlp_z)
            else:
                image_embeds = self.vision_encoder(image,head_z=vision_head_z,mlp_z=vision_mlp_z)[0]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.pad_token_id, -100)
            question_output = self.text_encoder(quesiton.input_ids,
                                                attention_mask=quesiton.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                output_attentions=output_attentions,
                                                output_hidden_states=output_hidden_states,
                                                head_z=torch.cat((text_head_z,cross_head_z),dim=0),
                                                mlp_z=torch.cat((text_mlp_z,cross_mlp_z),dim=0))
            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [quesiton.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              output_attentions=output_attentions,
                                              output_hidden_states=output_hidden_states,
                                              head_z=decoder_head_z,
                                              mlp_z=decoder_mlp_z
                                              )
            if output_attentions:
                decoder_logits,decoder_self,decoder_cross,decoder_hidden = answer_output.logits,answer_output.attentions,answer_output.cross_attentions,answer_output.hidden_states
                text_hidden_states,text_attentions,text_cross_attentions = question_output.hidden_states, question_output.attentions,question_output.cross_attentions

                hidden_dict = {
                'image_hidden_states':image_hidden_states,
                'text_hidden_states':text_hidden_states,
                'decoder_hidden_states':decoder_hidden
                    }
                attention_dict = {
                    'image_attentions':image_attentions,
                    'text_attentions':text_attentions,
                    'decoder_attentions':decoder_self
                }
                cross_attention_dict= {}
                logits_dict = {}
                logits_dict['logits'] = decoder_logits
                cross_attention_dict['cross_attentions'] = text_cross_attentions
                cross_attention_dict['decoder_cross_attentions'] = decoder_cross
                
                loss = weights * answer_output.loss
                loss = loss.sum() / image.size(0)
                
                res = {
                    'loss':loss,
                    'hidden_dict':hidden_dict,
                    'attention_dict':attention_dict,
                    'cross_attention_dict':cross_attention_dict,
                    'logits_dict':logits_dict
                }
                return res
            else:
                loss = weights * answer_output.loss
                loss = loss.sum() / image.size(0)
                return loss

        else:
            with torch.no_grad():
                zs = self.l0_module.forward(training=False)
            vision_head_z = zs['vision_head_z']
            vision_mlp_z = zs['vision_intermediate_z']
            text_head_z = zs['text_head_z']
            text_mlp_z = zs['text_intermediate_z']
            cross_head_z = zs['cross_head_z']
            cross_mlp_z = zs['cross_intermediate_z']
            decoder_head_z = zs['decoder_head_z']
            decoder_mlp_z = zs['decoder_intermediate_z']

            image_embeds = self.vision_encoder(image,head_z=vision_head_z,mlp_z=vision_mlp_z)[0]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            
            question_output = self.text_encoder(quesiton.input_ids,
                                                attention_mask=quesiton.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                head_z=torch.cat((text_head_z,cross_head_z),dim=0),
                                                mlp_z=torch.cat((text_mlp_z,cross_mlp_z),dim=0))
            topk_ids, topk_probs,_ = self.rank_answer(question_output.last_hidden_state, quesiton.attention_mask,
                                                    answer.input_ids, answer.attention_mask, k,[decoder_head_z,decoder_mlp_z])
            return topk_ids, topk_probs
            
    def fake_forward(self, image, quesiton, answer=None, k=None, weights=None, train=False,output_attentions=None,output_hidden_states=None,stop_prune=False):
        total_time = 0
        start_time = time.time()
        image_embeds = self.vision_encoder(image)[0]
        total_time += time.time() - start_time
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        start_time = time.time()
        question_output = self.text_encoder(quesiton.input_ids,
                                            attention_mask=quesiton.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True)
        total_time += time.time() - start_time
        topk_ids, topk_probs, decoder_time = self.rank_answer(question_output.last_hidden_state, quesiton.attention_mask,
                                                answer.input_ids, answer.attention_mask, k)
        return topk_ids, topk_probs ,total_time + decoder_time


    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k, decoder_mask=None):
        if decoder_mask is not None:
            decoder_head_z,decoder_mlp_z = decoder_mask
        else:
            decoder_head_z, decoder_mlp_z = None,None
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token
        total_time = 0
        start_time = time.time()
        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none',
                                         head_z=decoder_head_z,
                                         mlp_z=decoder_mlp_z)
        total_time += time.time() - start_time
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        start_time = time.time()
        output = self.text_decoder(input_ids,
                                    attention_mask=input_atts,
                                    encoder_hidden_states=question_states,
                                    encoder_attention_mask=question_atts,
                                    labels=targets_ids,
                                    return_dict=True,
                                    reduction='none',
                                    head_z=decoder_head_z,
                                    mlp_z=decoder_mlp_z)
        total_time += time.time() - start_time

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs, total_time


class EffXVLMForCaptioning(XVLMBase):  # generation based on images
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False, config_text=None)

        if config['text_encoder'] != 'data/bert-base-uncased':
            raise Warning(f"### for other text encoders ({config['text_encoder']}), not debug yet!")

        self.tokenizer = build_tokenizer(config['text_encoder'])
        self.tokenizer.add_special_tokens({'bos_token': self.tokenizer.cls_token, 'eos_token': self.tokenizer.sep_token})

        self.prompt = config['prompt']
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
        self.max_tokens = config['max_tokens']

        config_enc = self.text_encoder.config

        self.text_encoder = None
        self.text_decoder = BertLMHeadModel(config=config_enc, label_smoothing=config['label_smoothing'])
        self.l0_module = XVLML0Module(config,target_sparsity=config['sparsity'])

    def load_pretrained(self, ckpt_rpath, config, load_capt_pretrain=False, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(ckpt_rpath, config, load_text=False)
            print("### Loading pretrained text encoder", flush=True)
            print("load_capt_pretrain, ", load_capt_pretrain)
            if not load_capt_pretrain:
                print("### Loading pretrained text encoder", flush=True)
                for key in list(state_dict.keys()):
                    assert isinstance(key, str)
                    if key.startswith('text_encoder.'):
                        decoder_key = key.replace('text_encoder.', 'text_decoder.')
                        state_dict[decoder_key] = state_dict[key]
                        del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, caption,output_attentions=None,output_hidden_states=None):
        
        zs = self.l0_module.forward(training=True)   
        vision_head_z = zs['vision_head_z']
        vision_mlp_z = zs['vision_intermediate_z']
        text_head_z = zs['text_head_z']
        text_mlp_z = zs['text_intermediate_z']
        cross_head_z = zs['cross_head_z']
        cross_mlp_z = zs['cross_intermediate_z'] 
        if output_attentions:
            image_embeds,image_hidden_states,image_attentions = self.vision_encoder(image,
                                                                    output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                                                    head_z=vision_head_z,mlp_z=vision_mlp_z)
        else:
            image_embeds = self.vision_encoder(image)[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=self.max_tokens, return_tensors="pt").to(
            image.device)

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100
        outputs = self.text_decoder(text.input_ids,
                                 attention_mask=text.attention_mask,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 labels=decoder_targets,
                                 return_dict=True,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 head_z=torch.cat((text_head_z,cross_head_z),dim=0),
                                 mlp_z=torch.cat((text_mlp_z,cross_mlp_z),dim=0))
        if output_attentions:
            decoder_logits,decoder_self,decoder_cross,decoder_hidden = outputs.logits,outputs.attentions,outputs.cross_attentions,outputs.hidden_states
            loss = outputs.loss
            hidden_dict = {
                'image_hidden_states':image_hidden_states,
                'decoder_hidden_states':decoder_hidden
                    }
            attention_dict = {
                'image_attentions':image_attentions,
                'decoder_attentions':decoder_self
            }
            cross_attention_dict= {}
            logits_dict = {}
            logits_dict['logits'] = decoder_logits
            cross_attention_dict['decoder_cross_attentions'] = decoder_cross
            res = {
                    'loss':loss,
                    'hidden_dict':hidden_dict,
                    'attention_dict':attention_dict,
                    'cross_attention_dict':cross_attention_dict,
                    'logits_dict':logits_dict
                }
            return res
        
        else:
            return outputs.loss
    def generate(self, image, sample=False, num_beams=1, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0, num_return_sequences=1, greedy=False):
        total_time = 0
        with torch.no_grad():
            zs = self.l0_module.forward(training=False)
        vision_head_z = zs['vision_head_z']
        vision_mlp_z = zs['vision_intermediate_z']
        text_head_z = zs['text_head_z']
        text_mlp_z = zs['text_intermediate_z']
        cross_head_z = zs['cross_head_z']
        cross_mlp_z = zs['cross_intermediate_z']

        prompt = [self.prompt] * image.size(0)

        image_embeds = self.vision_encoder(image,head_z=vision_head_z,mlp_z=vision_mlp_z)[0]

        if num_beams > 1:
            assert (sample is False) and (num_return_sequences == 1)
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        if num_return_sequences > 1:
            assert (sample is True) and (num_beams == 1)
            image_embeds = image_embeds.repeat_interleave(num_return_sequences, dim=0)
            prompt = [self.prompt] * image_embeds.size(0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        # input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        def _get_captions(caption_ids):
            captions = []
            for output in caption_ids:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[len(self.prompt):])
            return captions

        if greedy:
            # greedy generation from OSCAR
            assert (num_beams == 1) and (num_return_sequences == 1)
            outputs, logprobs = self.text_decoder._generate_no_beam_search(input_ids=input_ids, cur_len=input_ids.shape[1], max_length=max_length,
                                          do_sample=False, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          pad_token_id=self.tokenizer.pad_token_id, eos_token_ids=[self.tokenizer.sep_token_id],
                                        #   head_z=torch.cat((text_head_z,cross_head_z),dim=0),
                                        #   mlp_z=torch.cat((text_mlp_z,cross_mlp_z),dim=0),
                                          batch_size=image_embeds.size(0), **model_kwargs)

            return _get_captions(outputs)

        elif sample:
            # sampling from OSCAR
            outputs, logprobs = self.text_decoder._generate_no_beam_search(input_ids=input_ids, cur_len=input_ids.shape[1], max_length=max_length,
                                          do_sample=True, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          pad_token_id=self.tokenizer.pad_token_id, eos_token_ids=[self.tokenizer.sep_token_id],
                                        #   head_z=torch.cat((text_head_z,cross_head_z),dim=0),
                                        #   mlp_z=torch.cat((text_mlp_z,cross_mlp_z),dim=0),
                                          batch_size=image_embeds.size(0), **model_kwargs)

            # outputs: (bs x num_return_sequences, max_length)
            # logprobs: (bs x num_return_sequences,)

            return _get_captions(outputs), logprobs

        else:
            # beam search from huggingface
            # start_time = time.time()
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 head_z=torch.cat((text_head_z,cross_head_z),dim=0),
                                                 mlp_z=torch.cat((text_mlp_z,cross_mlp_z),dim=0),
                                                 **model_kwargs)
            # total_time = time.time() - start_time
            return _get_captions(outputs)