import pdb

from re import L

import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from transformers.utils import logging
from transformers import BertConfig
import os
from utils import read_json
limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
logger = logging.get_logger(__name__)

class VQAL0Module(Module):
    def __init__(self,
                 config, 
                 droprate_init=0.5,
                 temperature=2./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 pruning_type="structured_heads+structured_mlp",
                 magical_number=0.8, # from Wang et al. 2020
                 ):
        super(VQAL0Module, self).__init__()
        #TODO: only support CLIP-VIT & BERT
        text_config = BertConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))
        text_config.num_hidden_layers = config['text_num_hidden_layers'] if 'text_num_hidden_layers' in config else 12
        assert  text_config.num_hidden_layers in [6, 12], "param initialization not implemented"
        text_config.fusion_layer = text_config.num_hidden_layers // 2
        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        self.all_types = ["vision_intermediate_z", "vision_head_z","text_intermediate_z", "text_head_z","cross_intermediate_z", "cross_head_z","decoder_head_z","decoder_intermediate_z"]
        self.pruning_type = pruning_type
        self.hidden_size = text_config.hidden_size
        self.intermediate_size = text_config.intermediate_size
        self.num_attention_heads = text_config.num_attention_heads
        self.dim_per_head = self.hidden_size // self.num_attention_heads 
        self.vision_num_hidden_layers = vision_config['num_hidden_layers']
        self.text_num_hidden_layers = text_config.fusion_layer
        self.cross_num_hidden_layers = text_config.num_hidden_layers - text_config.fusion_layer
        self.decoder_num_hidden_layers = self.cross_num_hidden_layers
        self.mlp_num_per_layer = 1
        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads       
        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 2 + self.hidden_size + self.hidden_size * 4
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size
        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.vision_num_hidden_layers + \
                                (self.params_per_head_layer + self.params_per_mlp_layer) * self.text_num_hidden_layers + \
                                (self.params_per_head_layer  * 2 + self.params_per_mlp_layer) * self.cross_num_hidden_layers + \
                                (self.params_per_head_layer  * 2 + self.params_per_mlp_layer) * self.decoder_num_hidden_layers
        
        self.prunable_model_size = 0 
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None

        types = self.pruning_type.split("+")
        for type in types:
            if type != "layer":
                self.initialize_one_module(type)
        if "layer" in types:
            self.initialize_one_module("layer")
        

        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        print("********** Initializing L0 Module **********") 
        for type in self.types:
            print(f"***** {type} *****")
            print(f"z.shape", self.z_logas[type].shape)
            print(f"size", self.sizes[type])
        print(f"prunable model size: {self.prunable_model_size}")

    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

    def initialize_one_module(self, module_name):
        if module_name == "structured_heads":
            self.initialize_structured_head()
        elif module_name == "structured_mlp":
            self.initialize_structured_mlp()
        
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape):
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape

    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))  

    def initialize_structured_head(self, add_prunable_model_size=True):
        self.vision_head_loga = self.initialize_parameters(self.num_attention_heads, self.vision_num_hidden_layers)
        self.text_head_loga = self.initialize_parameters(self.num_attention_heads, self.text_num_hidden_layers)
        self.cross_head_loga = self.initialize_parameters(self.num_attention_heads, self.cross_num_hidden_layers * 2)
        self.decoder_head_loga = self.initialize_parameters(self.num_attention_heads, self.decoder_num_hidden_layers * 2)
        self.reset_loga(self.vision_head_loga, mean=10)
        self.reset_loga(self.text_head_loga, mean=10)
        self.reset_loga(self.cross_head_loga, mean=10)
        self.reset_loga(self.decoder_head_loga, mean=10)

        self.add_one_module(self.vision_head_loga, type="vision_head", 
                            parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                            shape=[self.vision_num_hidden_layers, 1, self.num_attention_heads, 1, 1])
        self.add_one_module(self.text_head_loga, type="text_head", 
                            parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                            shape=[self.text_num_hidden_layers, 1, self.num_attention_heads, 1, 1])
        self.add_one_module(self.cross_head_loga, type="cross_head", 
                            parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                            shape=[self.cross_num_hidden_layers * 2, 1, self.num_attention_heads, 1, 1])
        self.add_one_module(self.decoder_head_loga, type="decoder_head", 
                            parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                            shape=[self.decoder_num_hidden_layers * 2, 1, self.num_attention_heads, 1, 1])
        if add_prunable_model_size:
            self.prunable_model_size += self.params_per_head * self.vision_num_hidden_layers * self.num_attention_heads
            self.prunable_model_size += self.params_per_head * self.text_num_hidden_layers * self.num_attention_heads
            self.prunable_model_size += self.params_per_head * self.cross_num_hidden_layers * 2 * self.num_attention_heads
            self.prunable_model_size += self.params_per_head * self.decoder_num_hidden_layers * 2 * self.num_attention_heads 
        print(f"Initialized structured heads! Prunable_model_size = {self.prunable_model_size}")


    def initialize_structured_mlp(self):
        self.vision_int_loga = self.initialize_parameters(self.intermediate_size, self.vision_num_hidden_layers)
        self.text_int_loga = self.initialize_parameters(self.intermediate_size, self.text_num_hidden_layers)
        self.cross_int_loga = self.initialize_parameters(self.intermediate_size, self.cross_num_hidden_layers)
        self.decoder_int_loga = self.initialize_parameters(self.intermediate_size, self.decoder_num_hidden_layers)

        self.add_one_module(self.vision_int_loga, type="vision_intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.vision_num_hidden_layers, 1, 1, self.intermediate_size])
        self.prunable_model_size += self.params_per_mlp_layer * self.vision_num_hidden_layers
        self.add_one_module(self.text_int_loga, type="text_intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.text_num_hidden_layers, 1, 1, self.intermediate_size])
        self.prunable_model_size += self.params_per_mlp_layer * self.text_num_hidden_layers
        self.add_one_module(self.cross_int_loga, type="cross_intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.cross_num_hidden_layers, 1, 1, self.intermediate_size])
        self.prunable_model_size += self.params_per_mlp_layer * self.cross_num_hidden_layers
        self.add_one_module(self.decoder_int_loga, type="decoder_intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.decoder_num_hidden_layers, 1, 1, self.intermediate_size])
        self.prunable_model_size += self.params_per_mlp_layer * self.decoder_num_hidden_layers
        self.reset_loga(self.vision_int_loga)
        self.reset_loga(self.text_int_loga)
        self.reset_loga(self.cross_int_loga)
        self.reset_loga(self.decoder_int_loga)
        print(f"Initialized structured mlp! Prunable_model_size = {self.prunable_model_size}")


    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)


    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])

    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size

    def transform_scores_for_head(self,modal):
        if modal == "vision":
            head_score = 1 - self.cdf_qz(0, self.vision_head_loga) # 12 * 12
        elif modal == "text":
            head_score = 1 - self.cdf_qz(0, self.text_head_loga)
        elif modal == "cross":
            head_score = 1 - self.cdf_qz(0, self.cross_head_loga)
        elif modal == 'decoder':
            head_score = 1 - self.cdf_qz(0, self.decoder_head_loga)
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1     
        return head_score


    def get_num_parameters_and_constraint(self):
        num_parameters = 0

        vision_head_score = self.transform_scores_for_head(modal='vision')
        text_head_score = self.transform_scores_for_head(modal='text')
        cross_head_score = self.transform_scores_for_head(modal='cross')
        decoder_head_score = self.transform_scores_for_head(modal='decoder')

        num_parameters += torch.sum(vision_head_score) * self.parameters_per_dim["vision_head"]
        num_parameters += torch.sum(text_head_score) * self.parameters_per_dim["text_head"]
        num_parameters += torch.sum(cross_head_score) * self.parameters_per_dim["cross_head"]
        num_parameters += torch.sum(decoder_head_score) * self.parameters_per_dim["decoder_head"]

        vision_int_score = 1 - self.cdf_qz(0, self.vision_int_loga)  # 12 * 3072
        text_int_score = 1 - self.cdf_qz(0, self.text_int_loga)
        cross_int_score = 1 - self.cdf_qz(0, self.cross_int_loga)
        decoder_int_score = 1 - self.cdf_qz(0, self.decoder_int_loga)

        num_parameters += torch.sum(vision_int_score) * self.parameters_per_dim["vision_intermediate"]
        num_parameters += torch.sum(text_int_score) * self.parameters_per_dim["text_intermediate"]
        num_parameters += torch.sum(cross_int_score) * self.parameters_per_dim["cross_intermediate"]
        num_parameters += torch.sum(decoder_int_score) * self.parameters_per_dim["decoder_intermediate"]
        return num_parameters


    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.target_sparsity
        expected_size = self.get_num_parameters_and_constraint()
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        lagrangian_loss = (
                self.lambda_1 * (expected_sparsity - target_sparsity)
                + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2
        )
        return lagrangian_loss, expected_sparsity, target_sparsity

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    # during inference
    def _deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask = torch.ones_like(soft_mask)
                soft_mask[indices] = 0.
        else:
            return torch.ones_like(soft_mask)
        return soft_mask


    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.all_types:
            name = type[:-2]
            z = zs.get(type, np.ones(self.shapes[name]))
            if torch.is_tensor(z): 
                new_z = z.squeeze().detach().cpu().numpy() > 0
            numpified_zs[name] = new_z
        return numpified_zs

    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        vision_intermediate_z = numpified_zs["vision_intermediate"]
        text_intermediate_z = numpified_zs["text_intermediate"]
        cross_intermediate_z = numpified_zs["cross_intermediate"]
        decoder_intermediate_z = numpified_zs["decoder_intermediate"]
        vision_head_z = numpified_zs["vision_head"]
        text_head_z = numpified_zs["text_head"]
        cross_head_z = numpified_zs["cross_head"]
        decoder_head_z = numpified_zs["decoder_head"]

        remaining_vision_intermediate_nums = vision_intermediate_z.reshape(self.vision_num_hidden_layers, self.intermediate_size).sum(-1).tolist()
        remaining_text_intermediate_nums = text_intermediate_z.reshape(self.text_num_hidden_layers, self.intermediate_size).sum(-1).tolist()
        remaining_cross_intermediate_nums = cross_intermediate_z.reshape(self.cross_num_hidden_layers, self.intermediate_size).sum(-1).tolist()
        remaining_decoder_intermediate_nums = decoder_intermediate_z.reshape(self.decoder_num_hidden_layers, self.intermediate_size).sum(-1).tolist()
        remaining_vision_head_nums = vision_head_z.reshape(self.vision_num_hidden_layers, self.num_attention_heads).sum(-1).tolist()
        remaining_text_head_nums = text_head_z.reshape(self.text_num_hidden_layers, self.num_attention_heads).sum(-1).tolist()
        remaining_cross_head_nums = cross_head_z.reshape(self.cross_num_hidden_layers * 2, self.num_attention_heads).sum(-1).tolist()
        remaining_decoder_head_nums = decoder_head_z.reshape(self.decoder_num_hidden_layers * 2, self.num_attention_heads).sum(-1).tolist()

        head_nums = sum(remaining_cross_head_nums) +sum(remaining_text_head_nums) + sum(remaining_vision_head_nums) + sum(remaining_decoder_head_nums)

        intermediate_nums = sum(remaining_vision_intermediate_nums) + sum(remaining_text_intermediate_nums) + \
                            sum(remaining_cross_intermediate_nums) + sum(remaining_decoder_intermediate_nums)

        remaining_model_size = head_nums * self.params_per_head + intermediate_nums * 2 * self.hidden_size
        pruned_model_size = self.prunable_model_size - remaining_model_size

        results = {}
        results["vision_intermediate_dims"] = remaining_vision_intermediate_nums
        results["text_intermediate_dims"] = remaining_text_intermediate_nums
        results["cross_intermediate_dims"] = remaining_cross_intermediate_nums
        results["decoder_intermediate_dims"] = remaining_decoder_intermediate_nums
        results["vision_head_nums"] = remaining_vision_head_nums
        results["text_head_nums"] = remaining_text_head_nums
        results["cross_head_nums"] = remaining_cross_head_nums
        results["decoder_head_nums"] = remaining_decoder_head_nums
        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size

        return results


    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                loga_all_layers = self.z_logas[type]
                for layer in range(len(loga_all_layers)):
                    loga = loga_all_layers[layer]
                    size = self.sizes[type]
                    z = self._deterministic_z(size, loga)
                    zs[f"{type}_z"].append(z.reshape(self.shapes[type][1:]))
            for type in zs:
                if type != "hidden_z":
                    zs[type] = torch.stack(zs[type])

        return zs 

