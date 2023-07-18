from efficient_models.xvlm import XVLMBase, load_pretrained
from efficient_models.xvlm_l0_module import XVLML0Module
from utils import read_json
import torch
class XVLMForGroundingPretraining(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=True)
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=False, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, idx_to_group_img, target_bbox, is_image=None):
        image_embeds_fullatts, _ = self.get_vision_embeds(image, idx_to_group_img=idx_to_group_img)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        output_coord = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts)
        loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)

        return loss_bbox, loss_giou



class EffXVLMForGrounding(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=True)
        self.init_params = []
        self.l0_module = XVLML0Module(config,target_sparsity=config['sparsity'])

    def load_pretrained(self, ckpt_rpath, config, load_bbox_pretrain=False, is_eval=False):
        print("### load_bbox_pretrain, ", load_bbox_pretrain)
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)
    

    def forward(self, image, text_ids, text_atts, target_bbox=None,output_attentions=None,output_hidden_states=None):
        train=False
        if target_bbox is not None:
            train=True
        if train:
            zs = self.l0_module.forward(training=train)   
        else:
            with torch.no_grad():
                zs = self.l0_module.forward(training=train)   
        vision_head_z = zs['vision_head_z']
        vision_mlp_z = zs['vision_intermediate_z']
        text_head_z = zs['text_head_z']
        text_mlp_z = zs['text_intermediate_z']
        cross_head_z = zs['cross_head_z']
        cross_mlp_z = zs['cross_intermediate_z']
        if output_attentions:
            image_embeds, _, image_hidden_states,image_attentions = self.get_vision_embeds(image,
                                                                    output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                                                    head_z=vision_head_z,mlp_z=vision_mlp_z)
            text_embeds,text_hidden_states,text_attentions = self.get_text_embeds(text_ids, text_atts,
                                                                    output_attentions=output_attentions,output_hidden_states=output_hidden_states,
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

            bbox_output = self.predict_bbox(image_embeds, text_embeds, text_atts,
                                            output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                                            head_z=cross_head_z,mlp_z=cross_mlp_z)
            output_coord = bbox_output[0]
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox)
            loss = {"loss_bbox":loss_bbox,
                    "loss_giou":loss_giou}
            bbox_hidden_states, bbox_attentions,bbox_cross_attentions = bbox_output[1:]
            hidden_dict['bbox_hidden_states'] = bbox_hidden_states
            attention_dict['bbox_attentions'] = bbox_attentions
            cross_attention_dict['bbox_cross_attentions'] = bbox_cross_attentions

            res = {
                'loss':loss,
                'hidden_dict':hidden_dict,
                'attention_dict':attention_dict,
                'cross_attention_dict':cross_attention_dict,
            }
        
            return res
        else:
            image_embeds, _ = self.get_vision_embeds(image,head_z=vision_head_z,mlp_z=vision_mlp_z)
            text_embeds = self.get_text_embeds(text_ids, text_atts,head_z=text_head_z,mlp_z=text_mlp_z)

            output_coord = self.predict_bbox(image_embeds, text_embeds, text_atts,head_z=cross_head_z,mlp_z=cross_mlp_z)[0]
            # output_coord & target_bbox: 64, 4

            if target_bbox is None:
                return output_coord

            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox)

            return output_coord, loss_bbox, loss_giou

