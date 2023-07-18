import torch
from models import XVLMBase


class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=True, config_text=None)


    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None, ret_bbox_loss=False,
                output_attentions=None,output_hidden_states=None):
        assert output_attentions == output_hidden_states
        
        if ret_bbox_loss:
            image_embeds, image_atts, image_embeds_fullatts,image_hidden_states,image_attentions= \
                self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img,output_attentions=output_attentions,output_hidden_states=output_hidden_states)
        else:
            image_embeds, image_atts,image_hidden_states,image_attentions = self.get_vision_embeds(image,output_attentions=output_attentions,output_hidden_states=output_hidden_states)

        text_embeds,text_hidden_states,text_attentions = self.get_text_embeds(text_ids, text_atts,output_attentions=output_attentions,output_hidden_states=output_hidden_states)

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

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat)
        itm_loss_dict = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat,
                                        output_attentions=output_attentions,output_hidden_states=output_hidden_states)
        loss_itm = itm_loss_dict['loss']
        hidden_dict['itm_pos_hidden_states'] = itm_loss_dict['pos_hidden_states']
        hidden_dict['itm_neg_hidden_states'] = itm_loss_dict['neg_hidden_states']
        attention_dict['itm_pos_attentions'] = itm_loss_dict['pos_attentions']
        attention_dict['itm_neg_attentions'] = itm_loss_dict['neg_attentions']
        cross_attention_dict['itm_pos_cross_attentions'] = itm_loss_dict['pos_cross_attentions']
        cross_attention_dict['itm_neg_cross_attentions'] = itm_loss_dict['neg_cross_attentions']
        logits_dict['itm_head_logits'] = itm_loss_dict['logits']

        mlm_loss_tuple = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids,
                                            output_attentions=output_attentions,output_hidden_states=output_hidden_states)
        loss_mlm = mlm_loss_tuple[0]
        hidden_dict['mlm_hidden_states'] = mlm_loss_tuple[2]
        attention_dict['mlm_attentions'] = mlm_loss_tuple[3]
        logits_dict['mlm_logits'] = mlm_loss_tuple[1]
        cross_attention_dict['mlm_cross_attentions'] = mlm_loss_tuple[4]

        loss = {'loss_itc': loss_itc, 'loss_itm': loss_itm, 'loss_mlm': loss_mlm}

        if ret_bbox_loss:
            bbox_output = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts,output_attentions=output_attentions,output_hidden_states=output_hidden_states)
            output_coord = bbox_output[0]
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)

            loss['loss_bbox'] = loss_bbox
            loss['loss_giou'] = loss_giou
            if output_attentions is not None:
                bbox_hidden_states, bbox_attentions,bbox_cross_attentions = bbox_output[1:]
                hidden_dict['bbox_hidden_states'] = bbox_hidden_states
                attention_dict['bbox_attentions'] = bbox_attentions
                cross_attention_dict['bbox_cross_attentions'] = bbox_cross_attentions
        res = {
            'loss':loss,
            'hidden_dict':hidden_dict,
            'attention_dict':attention_dict,
            'cross_attention_dict':cross_attention_dict,
            'logits_dict':logits_dict
        }

        return res