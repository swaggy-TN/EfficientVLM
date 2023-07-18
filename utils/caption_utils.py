# from black import main
import torch
import os
from transformers.file_utils import TF_RETURN_INTRODUCTION
from transformers.modeling_utils import prune_linear_layer
from efficient_models.xvlm import XVLMBase
from utils.utils import calculate_parameters



def load_model_with_zs(model_path, model_class, zs=None):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = AutoConfig.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, config=config)
    p = os.path.join(model_path, "pytorch_model.bin")
    loaded_weights = torch.load(p, map_location="cpu")
    model.load_state_dict(loaded_weights)
    print(f"Load weights from {model_path}")

    update_params(model, zs)
    print(f"Model Size before pruning: {calculate_parameters(model)}")
    prune_model_with_z(zs, model)
    print(f"Model Size after pruning: {calculate_parameters(model)}")
    return model

def load_model(model_path, model_class, zs=None, num_labels=2):
    if zs is not None:
        model = load_model_with_zs(model_path, model_class, zs)
    else:
        # only and task name and model weights are accessible 
        model = load_pruned_model(model_path, model_class)
        print(f"Model Size: {calculate_parameters(model)}")
    return model



def update_params(model, zs,cross_layers=3):
    text_encoder = model.text_encoder
    vision_encoder = model.vision_encoder
    dims_per_head = 64
    vision_layers,text_layers,cross_layers = 6,3,cross_layers
    if "text_intermediate_z" in zs:
        for layer in range(text_layers):
            intermediate_z = zs["text_intermediate_z"][layer].cpu().squeeze().clone()
            text_encoder.encoder.layer[layer].output.dense.weight.data = text_encoder.encoder.layer[layer].output.dense.weight.data.mul(intermediate_z)
    
    if "text_head_z" in zs:
        for layer in range(text_layers):
            head_z = zs["text_head_z"][layer].cpu().squeeze().clone()
            head_z = torch.repeat_interleave(head_z, dims_per_head)
            text_encoder.encoder.layer[layer].attention.self.value.weight.data = text_encoder.encoder.layer[layer].attention.self.value.weight.transpose(0, 1).data.mul(head_z).transpose(0, 1)
            text_encoder.encoder.layer[layer].attention.self.value.bias.data = text_encoder.encoder.layer[layer].attention.self.value.bias.data.mul(head_z)

    if "vision_intermediate_z" in zs:
        for layer in range(vision_layers):
            intermediate_z = zs["vision_intermediate_z"][layer].cpu().squeeze().clone()
            vision_encoder.encoder.layers[layer].mlp.fc2.weight.data = vision_encoder.encoder.layers[layer].mlp.fc2.weight.data.mul(intermediate_z)
    if "vision_head_z" in zs:
        for layer in range(vision_layers):
            head_z = zs["vision_head_z"][layer].cpu().squeeze().clone()
            head_z = torch.repeat_interleave(head_z, dims_per_head)
            vision_encoder.encoder.layers[layer].self_attn.v_proj.weight.data = vision_encoder.encoder.layers[layer].self_attn.v_proj.weight.transpose(0, 1).data.mul(head_z).transpose(0, 1)
            vision_encoder.encoder.layers[layer].self_attn.v_proj.bias.data = vision_encoder.encoder.layers[layer].self_attn.v_proj.bias.data.mul(head_z)

    if "cross_intermediate_z" in zs:
        for layer in range(cross_layers):
            intermediate_z = zs["cross_intermediate_z"][layer].cpu().squeeze().clone()
            text_encoder.encoder.layer[text_layers+layer].output.dense.weight.data = text_encoder.encoder.layer[text_layers+layer].output.dense.weight.data.mul(intermediate_z)
    if "cross_head_z" in zs:
        for layer in range(cross_layers):
            i = layer * 2
            self_head_z = zs["cross_head_z"][i].cpu().squeeze().clone()
            cross_head_z = zs["cross_head_z"][i+1].cpu().squeeze().clone()
            self_head_z = torch.repeat_interleave(self_head_z, dims_per_head)
            cross_head_z = torch.repeat_interleave(cross_head_z,dims_per_head)

            text_encoder.encoder.layer[text_layers+layer].attention.self.value.weight.data = text_encoder.encoder.layer[text_layers+layer].attention.self.value.weight.transpose(0, 1).data.mul(self_head_z).transpose(0, 1)
            text_encoder.encoder.layer[text_layers+layer].attention.self.value.bias.data = text_encoder.encoder.layer[text_layers+layer].attention.self.value.bias.data.mul(self_head_z)
            text_encoder.encoder.layer[text_layers+layer].crossattention.self.value.weight.data = text_encoder.encoder.layer[text_layers+layer].crossattention.self.value.weight.transpose(0, 1).data.mul(cross_head_z).transpose(0, 1)
            text_encoder.encoder.layer[text_layers+layer].crossattention.self.value.bias.data = text_encoder.encoder.layer[text_layers+layer].crossattention.self.value.bias.data.mul(cross_head_z)
    
     


def prune_model_with_z(zs, model,cross_layers=3):
    if zs is None:
        return None, None
    vision_encoder = model.vision_encoder
    text_encoder = model.text_encoder
    vision_layers,text_layers,cross_layers = 6,3,cross_layers
   
    if "vision_head_z" in zs:
        head_z = zs.get("vision_head_z", None)
        prune_heads = {}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index
        
            print(f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        vision_encoder.prune_heads(prune_heads)
    
    if "text_head_z" in zs:
        head_z = zs.get("text_head_z", None)
        prune_heads = {}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index
            print(f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        text_encoder.prune_heads(prune_heads)
    
    if "cross_head_z" in zs:    
        head_z = zs.get("cross_head_z", None)
        prune_heads = {}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index
            print(f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")    
        text_encoder.prune_heads(prune_heads,is_cross='cross')

    kept_intermediate_dims = None
    if "vision_intermediate_z" in zs:
        kept_intermediate_dims = {}
        intermediate_zs = zs["vision_intermediate_z"]
        for layer in range(len(intermediate_zs)):
            intermediate_z_layer = intermediate_zs[layer].squeeze()
            intermediate_z_layer = intermediate_z_layer.cpu().clone()
            kept_intermediate_dims[layer] = intermediate_z_layer.nonzero().reshape(-1).tolist()    
    if kept_intermediate_dims is not None:
        prune_vision_intermediate_layers(vision_encoder, kept_intermediate_dims,text_encoder.device)
    
    kept_intermediate_dims = None
    if "text_intermediate_z" in zs and "cross_intermediate_z" in zs:
        kept_intermediate_dims = {}
        text_intermediate_zs = zs["text_intermediate_z"]
        cross_intermediat_zs = zs["cross_intermediate_z"]
        intermediate_zs = torch.cat((text_intermediate_zs,cross_intermediat_zs),dim=0)
        for layer in range(len(intermediate_zs)):
            intermediate_z_layer = intermediate_zs[layer].squeeze()
            intermediate_z_layer = intermediate_z_layer.cpu().clone()
            kept_intermediate_dims[layer] = intermediate_z_layer.nonzero().reshape(-1).tolist()        
    if kept_intermediate_dims is not None:
        prune_intermediate_layers(text_encoder, kept_intermediate_dims,text_encoder.device)

#display vision encoder
    for layer in range(0, vision_layers):
        print("Layer:", layer)
        if vision_encoder.encoder.layers[layer].self_attn.q_proj is not None:
            print("query:", vision_encoder.encoder.layers[layer].self_attn.q_proj.weight.shape)
            print("key:", vision_encoder.encoder.layers[layer].self_attn.q_proj.weight.shape)
        else:
            print("query:", None)
            print("key:", None)
        if vision_encoder.encoder.layers[layer].self_attn.v_proj is not None:
            print("value:", vision_encoder.encoder.layers[layer].self_attn.v_proj.weight.shape)
            print("output:", vision_encoder.encoder.layers[layer].self_attn.out_proj.weight.shape)
        else:
            print("value:", None)
            print("output:", None)
        if vision_encoder.encoder.layers[layer].mlp.fc1 is not None:
            print("up:", vision_encoder.encoder.layers[layer].mlp.fc1.weight.shape)
            print("down:", vision_encoder.encoder.layers[layer].mlp.fc2.weight.shape)
        else:
            print("up", None)
            print("down", None) 
#display bert
    for layer in range(0, text_layers):
        print("Layer:", layer)
        if text_encoder.encoder.layer[layer].attention.self.query is not None:
            print("query:", text_encoder.encoder.layer[layer].attention.self.query.weight.shape)
            print("key:", text_encoder.encoder.layer[layer].attention.self.key.weight.shape)
        else:
            print("query:", None)
            print("key:", None)
        if text_encoder.encoder.layer[layer].attention.self.value is not None:
            print("value:", text_encoder.encoder.layer[layer].attention.self.value.weight.shape)
            print("output:", text_encoder.encoder.layer[layer].attention.output.dense.weight.shape)
        else:
            print("value:", None)
            print("output:", None)
        if text_encoder.encoder.layer[layer].intermediate.dense is not None:
            print("up:", text_encoder.encoder.layer[layer].intermediate.dense.weight.shape)
            print("down:", text_encoder.encoder.layer[layer].output.dense.weight.shape)
        else:
            print("up", None)
            print("down", None) 
#display cross_encoder           
    for layer in range(text_layers, text_layers+cross_layers):
        print("Layer:", layer)
        if text_encoder.encoder.layer[layer].attention.self.query is not None:
            print("self attention query:", text_encoder.encoder.layer[layer].attention.self.query.weight.shape)
            print("self attnetion key:", text_encoder.encoder.layer[layer].attention.self.key.weight.shape)
        else:
            print("self attention query:", None)
            print("self attnetion key:", None)
        if text_encoder.encoder.layer[layer].crossattention.self.query is not None:
            print("cross attention query:", text_encoder.encoder.layer[layer].crossattention.self.query.weight.shape)
            print("cross attnetion key:", text_encoder.encoder.layer[layer].crossattention.self.key.weight.shape)
        else:
            print("cross attention query:", None)
            print("cross attnetion key:", None)

        if text_encoder.encoder.layer[layer].attention.self.value is not None:
            print("self attention value:", text_encoder.encoder.layer[layer].attention.self.value.weight.shape)
            print("self attention output:", text_encoder.encoder.layer[layer].attention.output.dense.weight.shape)
        else:
            print("self attention value:", None)
            print("self attention output:", None)

        if text_encoder.encoder.layer[layer].crossattention.self.value is not None:
            print("cross attention value:", text_encoder.encoder.layer[layer].crossattention.self.value.weight.shape)
            print("cross attention output:", text_encoder.encoder.layer[layer].crossattention.output.dense.weight.shape)
        else:
            print("cross attention value:", None)
            print("cross attention output:", None)


        if text_encoder.encoder.layer[layer].intermediate.dense is not None:
            print("up:", text_encoder.encoder.layer[layer].intermediate.dense.weight.shape)
            print("down:", text_encoder.encoder.layer[layer].output.dense.weight.shape)
        else:
            print("up", None)
            print("down", None)


def prune_intermediate_layers(bert, keep_dims,device):
    for layer in keep_dims:
        if len(keep_dims[layer]) == 0:
            bert.encoder.layer[layer].intermediate.dense = None
            bert.encoder.layer[layer].output.dense = None
        else:
            bert.encoder.layer[layer].intermediate.dense = prune_linear_layer(bert.encoder.layer[layer].intermediate.dense, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
            bert.encoder.layer[layer].output.dense = prune_linear_layer(bert.encoder.layer[layer].output.dense, index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)

def prune_vision_intermediate_layers(vision_encoder, keep_dims,device):
    for layer in keep_dims:
        if len(keep_dims[layer]) == 0:
            vision_encoder.encoder.layers[layer].mlp.fc1 = None
            vision_encoder.encoder.layers[layer].mlp.fc2 = None
        else:
            vision_encoder.encoder.layers[layer].mlp.fc1 = prune_linear_layer(vision_encoder.encoder.layers[layer].mlp.fc1, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
            vision_encoder.encoder.layers[layer].mlp.fc2 = prune_linear_layer(vision_encoder.encoder.layers[layer].mlp.fc2, index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)
         

def load_zs(model_path):
    if model_path.endswith("zs.pt"):
        zs_path = model_path
    else:
        zs_path = os.path.join(model_path, "zs.pt")

    if os.path.exists(zs_path):
        zs = torch.load(zs_path, map_location="cpu")
        if zs is None:
            model_path = os.path.dirname(model_path)
            l0_module = torch.load(os.path.join(model_path, "l0_module.pt"), map_location="cpu")
            zs = l0_module.forward(training=False)
        return zs
    else:
        return None

def load_pruned_model(model, weights):
    config = model.config
    dim_per_head = config.hidden_size // config.num_attention_heads
    zs = {}

    architecture = config.architectures[0].lower()
    bert_name = "roberta" if "roberta" in architecture else "bert"
    
    hidden_z = torch.zeros(config.hidden_size)
    hidden_z[:weights[f"{bert_name}.embeddings.word_embeddings.weight"].shape[1]] = 1
    zs["hidden_z"] = hidden_z

    head_z = torch.zeros(config.num_hidden_layers, config.num_attention_heads)    
    head_layer_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"{bert_name}.encoder.layer.{i}.attention.output.dense.weight"
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head 
            head_z[i, :remaining_heads] = 1
            head_layer_z[i] = 1
    zs["head_z"] = head_z
    zs["head_layer_z"] = head_layer_z

    int_z = torch.zeros(config.num_hidden_layers, config.intermediate_size)
    mlp_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"bert.encoder.layer.{i}.output.dense.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1] 
            int_z[i, :remaining_int_dims] = 1
            mlp_z[i] = 1
    zs["intermediate_z"] = int_z
    zs["mlp_z"] = mlp_z
    
    prune_model_with_z(zs, model)    
    model.load_state_dict(weights, strict=False)
    return model

def get_full_model_size(model_class, model_name):
    model = model_class.from_pretrained(model_name) 
    model_size = calculate_parameters(model)
    return model_size


