In this part of demo, we try to apply dynamic Lora into stable diffusion 1.5 model without recompile model. Since we do not have related API, such as: modify_weights, add_weights and so on, all lora weights inputs will be treated as Parameter input of whole model. Here is an example for a part of new lora parameter input:
![alt text](example_graph_with_lora-1.png)

# 1. Performance
see attachment file **performance.xlsx**

# 2. Code 
```bash
python ov_model_export.py
```

```bash
python ov_model_infer.py
```

# 2. Code Explaination
The most important part is the code in **util.py**, which is used to modify the model graph and load lora.

Function *load_lora(lora_path, DEVICE_NAME)* is used to load lora, get lora's shape and weights per layers and modify each layer's name. 

```python
def load_lora(lora_path, DEVICE_NAME):
    state_dict = load_file(lora_path)
    if DEVICE_NAME =="CPU":
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                    value_fp32 = value.type(torch.float32)
                    state_dict[key] = value_fp32

    layers_per_block = 2#TODO
    state_dict = _maybe_map_sgm_blocks_to_diffusers(state_dict, layers_per_block)
    state_dict, network_alphas = _convert_non_diffusers_lora_to_diffusers(state_dict)

    # now keys in format like: "unet.up_blocks.0.attentions.2.transformer_blocks.8.ff.net.2.lora.down.weight"'
    new_state_dict = {}
    for key , value in state_dict.items():
        if len(value.shape)==4:
            # new_value = torch.reshape(value, (value.shape[0],value.shape[1]))
            new_value = torch.squeeze(value)
        else:
            new_value = value
        new_state_dict[key.replace('.', '_').replace('_processor','')] = new_value
    # now keys in format like: "unet_up_blocks_0_attentions_2_transformer_blocks_8_ff_net_2_lora_down_weight"'

    LORA_PREFIX_UNET = "unet"
    LORA_PREFIX_TEXT_ENCODER = "text_encoder"
    LORA_PREFIX_TEXT_2_ENCODER = "text_encoder_2"

    lora_text_encoder_input_value_dict = {}
    lora_text_encoder_2_input_value_dict = {}
    lora_unet_input_value_dict = {}

    lora_alpha = collections.Counter(network_alphas.values()).most_common()[0][0]

    for key in new_state_dict.keys():
        if LORA_PREFIX_TEXT_ENCODER in key and "lora_down" in key and LORA_PREFIX_TEXT_2_ENCODER not in key:
            layer_infos = key.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1]
            lora_text_encoder_input_value_dict[layer_infos] = new_state_dict[key]
            lora_text_encoder_input_value_dict[layer_infos.replace("lora_down", "lora_up")] = new_state_dict[key.replace("lora_down", "lora_up")]

        elif LORA_PREFIX_TEXT_2_ENCODER in key and "lora_down" in key:
            layer_infos = key.split(LORA_PREFIX_TEXT_2_ENCODER + "_")[-1]
            lora_text_encoder_2_input_value_dict[layer_infos] = new_state_dict[key]
            lora_text_encoder_2_input_value_dict[layer_infos.replace("lora_down", "lora_up")] = new_state_dict[key.replace("lora_down", "lora_up")]

        elif LORA_PREFIX_UNET in key and "lora_down" in key:
            layer_infos = key.split(LORA_PREFIX_UNET + "_")[-1]
            lora_unet_input_value_dict[layer_infos] = new_state_dict[key]
            lora_unet_input_value_dict[layer_infos.replace("lora_down", "lora_up")] = new_state_dict[key.replace("lora_down", "lora_up")]

    #now the keys in format without prefix

    return lora_text_encoder_input_value_dict, lora_text_encoder_2_input_value_dict, lora_unet_input_value_dict, lora_alpha
```

Function add_param(model, lora_input_value_dict) is used to add input parameter per names of related layers, which will be connected to model with *manager.register_pass(InsertLoRAUnet(input_param_dict))* and *manager.register_pass(InsertLoRATE(input_param_dict))*, in these two classes, we search the whole model graph to find the related layers by their names and connect them with lora.

```python
def add_param(model, lora_input_value_dict):
        param_list = []
        for key, value in lora_input_value_dict.items():
            if '_lora_down' in key:
                key_down = key
                key_up = key_down.replace('_lora_down','_lora_up')
                name_alpha = key_down.replace('_lora_down','_lora_alpha')
                lora_alpha = ops.parameter(shape='',name=name_alpha)
                lora_alpha.output(0).set_names({name_alpha})
                lora_down = ops.parameter(shape=[-1, lora_input_value_dict[key_down].shape[-1]], name=key_down)
                lora_down.output(0).set_names({key_down})
                lora_up = ops.parameter(shape=[lora_input_value_dict[key_up].shape[0], -1], name=key_up)
                lora_up.output(0).set_names({key_up})
                param_list.append(lora_alpha)
                param_list.append(lora_down)
                param_list.append(lora_up)
        model.add_parameters(param_list)
```

```python
class InsertLoRAUnet(MatcherPass):
    def __init__(self, input_param_dict):
        MatcherPass.__init__(self)
        self.model_changed = False
        param = WrapType("opset10.Convert")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            root_output = matcher.get_match_value()
            for key in input_param_dict.keys():
                if root.get_friendly_name().replace('.','_').replace('self_unet_','') == key.replace('_lora_down','').replace('to_out','to_out_0'):

                    key_down = key
                    key_up = key_down.replace('_lora_down','_lora_up')
                    key_alpha = key_down.replace('_lora_down','_lora_alpha')

                    consumers = root_output.get_target_inputs()

                    lora_up_node = input_param_dict.pop(key_up)
                    lora_down_node = input_param_dict.pop(key_down)
                    lora_alpha_node = input_param_dict.pop(key_alpha)   

                    lora_weights = ops.matmul(data_a=lora_up_node, data_b=lora_down_node, transpose_a=False, transpose_b=False, name=key.replace('_down',''))
                    lora_weights_alpha = ops.multiply(lora_alpha_node, lora_weights)
                    if len(root.shape)!=len(lora_weights_alpha.shape):
                        # lora_weights_alpha_reshape = ops.reshape(lora_weights_alpha, root.shape, special_zero=False)
                        lora_weights_alpha_reshape = ops.unsqueeze(lora_weights_alpha, axes=[2, 3])
                        add_lora = ops.add(root,lora_weights_alpha_reshape,auto_broadcast='numpy')
                    else:
                        add_lora = ops.add(root,lora_weights_alpha,auto_broadcast='numpy')
                    for consumer in consumers:
                        consumer.replace_source_output(add_lora.output(0))

                    return True
            # Root node wasn't replaced or changed
            return False
        
        self.register_matcher(Matcher(param,"InsertLoRAUnet"), callback)
```
```python
class InsertLoRATE(MatcherPass):
    def __init__(self, input_param_dict):
        MatcherPass.__init__(self)
        self.model_changed = False
        param = WrapType("opset10.Convert")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            root_output = matcher.get_match_value()
            root_name = None
            if 'Constant_' in root.get_friendly_name() and root.shape == ov.Shape([768,768]):
                target_input = root.output(0).get_target_inputs()
                for v in target_input:
                    for input_of_MatMul in v.get_node().inputs():
                        if input_of_MatMul.get_shape()== ov.Shape([1,77,768]):
                            Add_Node = input_of_MatMul.get_source_output().get_node()
                            for Add_Node_output in Add_Node.output(0).get_target_inputs():
                                if 'k_proj' in Add_Node_output.get_node().get_friendly_name():
                                    for i in Add_Node_output.get_node().inputs():
                                        if i.get_shape() == ov.Shape([768,768]) and 'k_proj' in i.get_source_output().get_node().get_friendly_name():
                                            root_name = i.get_source_output().get_node().get_friendly_name().replace('k_proj', 'q_proj')

            root_friendly_name = root_name if root_name else root.get_friendly_name()
            
            for key in input_param_dict.keys():
                if root_friendly_name.replace('.','_').replace('self_','') == key.replace('_lora_down','_proj').replace('_to','').replace('_self',''):
                    # print(root_friendly_name)
                    key_down = key
                    key_up = key_down.replace('_lora_down','_lora_up')
                    key_alpha = key_down.replace('_lora_down','_lora_alpha')

                    consumers = root_output.get_target_inputs()

                    lora_up_node = input_param_dict.pop(key_up)
                    lora_down_node = input_param_dict.pop(key_down)
                    lora_alpha_node = input_param_dict.pop(key_alpha)   

                    lora_weights = ops.matmul(data_a=lora_up_node, data_b=lora_down_node, transpose_a=False, transpose_b=False, name=key.replace('_down',''))
                    lora_weights_alpha = ops.multiply(lora_alpha_node, lora_weights)
                    add_lora = ops.add(root,lora_weights_alpha,auto_broadcast='numpy')
                    for consumer in consumers:
                        consumer.replace_source_output(add_lora.output(0))

                    return True
                
            if len(input_param_dict) == 0:
                print("All loras are added")
            # Root node wasn't replaced or changed
            return False
        
        self.register_matcher(Matcher(param,"InsertLoRATE"), callback)
```