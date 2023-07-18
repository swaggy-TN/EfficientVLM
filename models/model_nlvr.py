import os
import torch
from torch import nn
import torch.nn.functional as F

from models import XVLMBase, build_mlp, load_pretrained
from models.xbert import BertConfig


class XVLMForNLVRPretraining(XVLMBase):
    def __init__(self, config):

        config_text = BertConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))
        config_text.num_hidden_layers = config['text_num_hidden_layers'] if 'text_num_hidden_layers' in config else 12
        assert config_text.num_hidden_layers in [6, 12], "param initialization not implemented"
        config_text.fusion_layer = config_text.num_hidden_layers // 2

        num_text_layers = config_text.fusion_layer
        num_cross_layers = config_text.num_hidden_layers - config_text.fusion_layer
        config_text.num_hidden_layers = num_text_layers + 2 * num_cross_layers

        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=config_text)
        self.num_text_layers = num_text_layers  # overwrite
        self.num_cross_layers = num_cross_layers

        # share the cross-attention layers for two images
        self.share_cross_attention(self.text_encoder.encoder)
        self.vision_proj = nn.Linear(self.vision_width, config['embed_dim'])
        self.ta_head = nn.Linear(self.text_width, 3)
        self.init_params = ['ta_head.' + n for n, _ in self.ta_head.named_parameters()]

    def load_pretrained(self, ckpt_rpath, config):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=False, load_text=False)

        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if 'text_encoder.' in key:
                if ('bert.' in key) or ('roberta.' in key):
                    new_key = key.replace('bert.', '')
                    new_key = new_key.replace('roberta.', '')

                    if 'layer.' in new_key:
                        keys = new_key.split('.')
                        layer_num = int(keys[3])
                        # replicate the multimodal encoder's blocks for two images
                        if layer_num >= self.num_text_layers:
                            new_layer_num = (layer_num - self.num_text_layers) * 2 + self.num_text_layers
                            keys[3] = str(new_layer_num)
                            new_key_0 = '.'.join(keys)
                            state_dict[new_key_0] = state_dict[key]
                            keys[3] = str(new_layer_num + 1)
                            new_key_1 = '.'.join(keys)
                            state_dict[new_key_1] = state_dict[key]
                        else:
                            state_dict[new_key] = state_dict[key]

                    else:
                        state_dict[new_key] = state_dict[key]

                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts):

        image_embeds, image_atts = self.get_vision_embeds(image)

        with torch.no_grad():
            image_feat = self.get_features(image_embeds=image_embeds)
            sim = image_feat @ image_feat.t() / 0.07
            weights = F.softmax(sim, dim=1)
            weights.fill_diagonal_(0)

        image_inputs = [[], []]
        labels = []
        for b in range(image.size(0)):
            if torch.rand(1) > 1 / 3:
                idx = torch.multinomial(weights[b], 1).item()
                if torch.rand(1) > 0.5:
                    image_inputs[0].append(image_embeds[b])
                    image_inputs[1].append(image_embeds[idx])
                    labels.append(0)
                else:
                    image_inputs[1].append(image_embeds[b])
                    image_inputs[0].append(image_embeds[idx])
                    labels.append(1)
            else:
                idx = torch.multinomial(weights[b], 2)
                image_inputs[0].append(image_embeds[idx[0]])
                image_inputs[1].append(image_embeds[idx[1]])
                labels.append(2)

        image_inputs[0] = torch.stack(image_inputs[0], dim=0)
        image_inputs[1] = torch.stack(image_inputs[1], dim=0)
        labels = torch.LongTensor(labels).to(image.device)

        output_cls = self.get_cross_embeds(image_inputs, [image_atts, image_atts],
                                           text_ids=text_ids, text_atts=text_atts)[:, 0, :]
        pred = self.ta_head(output_cls)
        loss = F.cross_entropy(pred, labels)

        return loss

    def share_cross_attention(self, model):
        for i in range(self.num_cross_layers):
            layer_num = self.num_text_layers + i * 2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num + 1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias



class XVLMForNLVR(XVLMBase):
    def __init__(self, config):
        config_text = BertConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))
        config_text.num_hidden_layers = config['text_num_hidden_layers'] if 'text_num_hidden_layers' in config else 12
        assert config_text.num_hidden_layers in [6, 12], "param initialization not implemented"
        config_text.fusion_layer = config_text.num_hidden_layers // 2

        num_text_layers = config_text.fusion_layer
        num_cross_layers = config_text.num_hidden_layers - config_text.fusion_layer
        config_text.num_hidden_layers = num_text_layers + 2 * num_cross_layers

        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=config_text)
        self.num_text_layers = num_text_layers  # overwrite
        self.num_cross_layers = num_cross_layers
        self.share_cross_attention(self.text_encoder.encoder)

        self.cls_head = build_mlp(input_dim=self.text_width, output_dim=2)
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def load_pretrained(self, ckpt_rpath, config, load_nlvr_pretrain=False, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(ckpt_rpath, config, load_text=False)
            print("### Loading pretrained text encoder", flush=True)
            print("load_nlvr_pretrain, ", load_nlvr_pretrain)
            if not load_nlvr_pretrain:
                for key in list(state_dict.keys()):
                    if 'text_encoder.' in key:
                        if ('bert.' in key) or ('roberta.' in key):
                            new_key = key.replace('bert.', '')
                            new_key = new_key.replace('roberta.', '')

                            if 'layer.' in new_key:
                                keys = new_key.split('.')
                                layer_num = int(keys[3])
                                # replicate the multimodal encoder's blocks for two images
                                if layer_num >= self.num_text_layers:
                                    new_layer_num = (layer_num - self.num_text_layers) * 2 + self.num_text_layers
                                    keys[3] = str(new_layer_num)
                                    new_key_0 = '.'.join(keys)
                                    state_dict[new_key_0] = state_dict[key]
                                    keys[3] = str(new_layer_num + 1)
                                    new_key_1 = '.'.join(keys)
                                    state_dict[new_key_1] = state_dict[key]
                                else:
                                    state_dict[new_key] = state_dict[key]

                            else:
                                state_dict[new_key] = state_dict[key]

                            del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, targets, train=True,output_attentions=None,output_hidden_states=None):
        if output_attentions:
            image_embeds, image_atts,image_hidden_states,image_attentions = self.get_vision_embeds(image,output_attentions=output_attentions,output_hidden_states=output_hidden_states)
            image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))
            outputs = self.get_cross_embeds([image0_embeds, image1_embeds], [image_atts[:image0_embeds.size(0)], image_atts[image0_embeds.size(0):]],
                                            text_ids=text_ids, text_atts=text_atts,output_attentions=output_attentions,output_hidden_states=output_hidden_states)
            output_cls = outputs[0][:,0,:]
            prediction = self.cls_head(output_cls)
            loss = F.cross_entropy(prediction, targets) if train else None

            text_hidden_states,text_attentions,cross_attentions = outputs[1:]
    
            hidden_dict = {
                'image_hidden_states':image_hidden_states,
                'text_hidden_states':text_hidden_states
            }
            attention_dict = {
                'image_attentions':image_attentions,
                'text_attentions':text_attentions
            }
            cross_attention_dict = {'cross_attentions':cross_attentions}
            logits_dict = {'cls_head_logits':prediction}

            res = {
                'loss':loss,
                'hidden_dict':hidden_dict,
                'attention_dict':attention_dict,
                'cross_attention_dict':cross_attention_dict,
                'logits_dict':logits_dict
            }
        
            return res
        else:
            image_embeds, image_atts = self.get_vision_embeds(image)
            image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))

            output_cls = self.get_cross_embeds([image0_embeds, image1_embeds], [image_atts[:image0_embeds.size(0)], image_atts[image0_embeds.size(0):]],
                                            text_ids=text_ids, text_atts=text_atts)[:, 0, :]

            prediction = self.cls_head(output_cls)

            return F.cross_entropy(prediction, targets) if train else prediction

    def share_cross_attention(self, model):
        for i in range(self.num_cross_layers):
            layer_num = self.num_text_layers + i * 2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num+1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias