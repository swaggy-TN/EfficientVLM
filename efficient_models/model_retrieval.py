import torch
from torch.nn.functional import triplet_margin_loss
from efficient_models.xvlm import XVLMBase, load_pretrained
from efficient_models.xvlm_l0_module import XVLML0Module


class EffXVLMforRetrieval(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False)

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.l0_module = XVLML0Module(config,target_sparsity=config['sparsity'])

        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)


    def forward(self, image, text_ids, text_atts, idx=None,output_attentions=None,output_hidden_states=None):
        if output_attentions:
            zs = self.l0_module.forward(training=True)
            vision_head_z = zs['vision_head_z']
            vision_mlp_z = zs['vision_intermediate_z']

            image_embeds, image_atts,image_hidden_states,image_attentions = self.get_vision_embeds(image,output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                                            head_z=vision_head_z,mlp_z=vision_mlp_z)

            text_head_z = zs['text_head_z']
            text_mlp_z = zs['text_intermediate_z']
            text_embeds,text_hidden_states,text_attentions = self.get_text_embeds(text_ids, text_atts,output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                                                                head_z=text_head_z,mlp_z=text_mlp_z)
            hidden_dict = {
                    'image_hidden_states':image_hidden_states,
                    'text_hidden_states':text_hidden_states
                }
            attention_dict = {
                'image_attentions':image_attentions,
                'text_attentions':text_attentions
            }
            cross_attention_dict = {}
            logits_dict = {}

            image_feat, text_feat = self.get_features(image_embeds, text_embeds)
            loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)

            cross_head_z = zs['cross_head_z']
            cross_mlp_z = zs['cross_intermediate_z']
            itm_loss_dict = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx,
                                                output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                                head_z=cross_head_z,mlp_z=cross_mlp_z)
            loss_itm = itm_loss_dict['loss']
                
            loss = {'loss_itc': loss_itc, 'loss_itm': loss_itm}
            hidden_dict['itm_pos_hidden_states'] = itm_loss_dict['pos_hidden_states']
            hidden_dict['itm_neg_hidden_states'] = itm_loss_dict['neg_hidden_states']
            attention_dict['itm_pos_attentions'] = itm_loss_dict['pos_attentions']
            attention_dict['itm_neg_attentions'] = itm_loss_dict['neg_attentions']
            cross_attention_dict['itm_pos_cross_attentions'] = itm_loss_dict['pos_cross_attentions']
            cross_attention_dict['itm_neg_cross_attentions'] = itm_loss_dict['neg_cross_attentions']
            logits_dict['itm_head_logits'] = itm_loss_dict['logits']

            res = {
                'loss':loss,
                'hidden_dict':hidden_dict,
                'attention_dict':attention_dict,
                'cross_attention_dict':cross_attention_dict,
                'logits_dict':logits_dict,
            }

            return res
        else:
            zs = self.l0_module.forward(training=False)
            vision_head_z = zs['vision_head_z']
            vision_mlp_z = zs['vision_intermediate_z']
            text_head_z = zs['text_head_z']
            text_mlp_z = zs['text_intermediate_z']
            cross_head_z = zs['cross_head_z']
            cross_mlp_z = zs['cross_intermediate_z']
            image_embeds, image_atts = self.get_vision_embeds(image,head_z=vision_head_z,mlp_z=vision_mlp_z)
            text_embeds = self.get_text_embeds(text_ids, text_atts,head_z=text_head_z,mlp_z=text_mlp_z)

            image_feat, text_feat = self.get_features(image_embeds, text_embeds)
            loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx,
                                            head_z=cross_head_z,mlp_z=cross_mlp_z)

            return loss_itc, loss_itm