
import os
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from pathlib import Path
import openvino as ov
import torch 
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial

DEVICE_NAME="GPU.1"
UNET_OV_PATH = Path("./ov_models_dynamic/unet/openvino_model.xml")
CONTROLNET_OV_PATH = Path("./ov_models_dynamic/controlnet/openvino_model.xml")
TEXT_ENCODER_OV_PATH = Path("./ov_models_dynamic/text_encoder/openvino_model.xml")
TOKENIZER_OV_PATH = Path("./ov_models_dynamic/tokenizer")
SCHEDULER_OV_PATH = Path("./ov_models_dynamic/scheduler")
VAE_DECODER_OV_PATH = Path("./ov_models_dynamic/vae_decoder/openvino_model.xml")

UNET_STATIC_OV_PATH = Path("./ov_models_static/unet/openvino_model.xml")
CONTROLNET_STATIC_OV_PATH = Path("./ov_models_static/controlnet/openvino_model.xml")
TEXT_ENCODER_STATIC_OV_PATH = Path("./ov_models_static/text_encoder/openvino_model.xml")
VAE_DECODER_STATIC_OV_PATH = Path("./ov_models_static/vae_decoder/openvino_model.xml")

NEED_STATIC = True
STATIC_SHAPE = [1024,1024]

USE_LORA=True
UNET_LORA_OV_PATH = Path("./ov_models_static/unet_lora/openvino_model.xml")
TEXT_ENCODER_LORA_OV_PATH = Path("./ov_models_static/text_encoder_lora/openvino_model.xml")

core = ov.Core()

if UNET_OV_PATH.exists() and CONTROLNET_OV_PATH.exists() and TEXT_ENCODER_OV_PATH.exists()  and VAE_DECODER_OV_PATH.exists(): #and TEXT_ENCODER_2_OV_PATH.exists()
    print("Loading OpenVINO models")
else:
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart",torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32)       

#config save
if not TOKENIZER_OV_PATH.exists():
    pipe.tokenizer.save_pretrained(TOKENIZER_OV_PATH)

if not SCHEDULER_OV_PATH.exists():
    pipe.scheduler.save_config(SCHEDULER_OV_PATH)

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


#controlnet
inputs = {
    "sample": torch.randn((2, 4, 64, 64)),
    "timestep": torch.tensor(1),
    "encoder_hidden_states": torch.randn((2, 77, 768)),
    "controlnet_cond": torch.randn((2, 3, 512, 512)),
}

input_info = []
for name, inp in inputs.items():
    shape = ov.PartialShape(inp.shape)
    # element_type = dtype_mapping[input_tensor.dtype]
    if len(shape) == 4:
        shape[0] = -1
        shape[2] = -1
        shape[3] = -1
    elif len(shape) == 3:
        shape[0] = -1
    input_info.append((shape))


if not CONTROLNET_OV_PATH.exists():
    controlnet.eval()
    with torch.no_grad():
        down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)
        controlnet.forward = partial(controlnet.forward, return_dict=False)
        ov_model = ov.convert_model(controlnet, example_input=inputs, input=input_info)
        ov.save_model(ov_model, CONTROLNET_OV_PATH)
        del ov_model
        cleanup_torchscript_cache()
    print("ControlNet successfully converted to IR")
    del controlnet
else:
    print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")

#unet
dtype_mapping = {
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
}


class UnetWrapper(torch.nn.Module):
    def __init__(
        self,
        unet,
        sample_dtype=torch.float32,
        timestep_dtype=torch.int64,
        encoder_hidden_states=torch.float32,
        down_block_additional_residuals=torch.float32,
        mid_block_additional_residual=torch.float32,
    ):
        super().__init__()
        self.unet = unet
        self.sample_dtype = sample_dtype
        self.timestep_dtype = timestep_dtype
        self.encoder_hidden_states_dtype = encoder_hidden_states
        self.down_block_additional_residuals_dtype = down_block_additional_residuals
        self.mid_block_additional_residual_dtype = mid_block_additional_residual

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: Tuple[torch.Tensor],
        mid_block_additional_residual: torch.Tensor,
    ):
        sample.to(self.sample_dtype)
        timestep.to(self.timestep_dtype)
        encoder_hidden_states.to(self.encoder_hidden_states_dtype)
        down_block_additional_residuals = [res.to(self.down_block_additional_residuals_dtype) for res in down_block_additional_residuals]
        mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        )
    
def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs

if not UNET_OV_PATH.exists():
    inputs.pop("controlnet_cond", None)
    
    inputs["down_block_additional_residuals"] = down_block_res_samples
    inputs["mid_block_additional_residual"] = mid_block_res_sample

    unet = UnetWrapper(pipe.unet)
    unet.eval()

    with torch.no_grad():
        ov_model = ov.convert_model(unet, example_input=inputs)
  
    flatten_inputs = flattenize_inputs(inputs.values())
    a = 1

    for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):
        r_name = input_tensor.get_node().get_friendly_name()
        r_shape = ov.PartialShape(input_data.shape)
        print("============")
        print(r_name, r_shape)
        
        if len(r_shape) == 4:
            r_shape[0] = -1
            r_shape[2] = -1
            r_shape[3] = -1
        elif len(r_shape) == 3:
            r_shape[0] = -1
            r_shape[1] = -1
        elif len(r_shape) == 2:
            r_shape[0] = -1
            r_shape[1] = -1
        tn = "down_block_additional_residual_"
        if r_name not in ["sample", "timestep", "encoder_hidden_states", "mid_block_additional_residual", "text_embeds", "time_ids"] and len(r_shape)==4:
            n_name = tn + str(a)
            if a == 23:
                n_name = "down_block_additional_residual"
            input_tensor.get_node().set_friendly_name(n_name)
            a = a + 2
        input_tensor.get_node().set_partial_shape(r_shape)
        input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])

    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, UNET_OV_PATH)
    del ov_model
    cleanup_torchscript_cache()
    del unet
    del pipe.unet
    print("Unet successfully converted to IR")
else:
    print(f"Unet will be loaded from {UNET_OV_PATH}")


def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
    """
    Convert Text Encoder model to OpenVINO IR.
    Function accepts text encoder model, prepares example inputs for conversion, and convert it to OpenVINO Model
    Parameters:
        text_encoder (torch.nn.Module): text_encoder model
        ir_path (Path): File for storing model
    Returns:
        None
    """
    if not ir_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.int64)
        # switch model to inference mode
        text_encoder.eval()
        text_encoder.config.output_hidden_states = True
        text_encoder.config.return_dict = False
        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            ov_model = ov.convert_model(
                text_encoder,  # model instance
                example_input=input_ids,  # inputs for model tracing
                input=([1, 77],),
            )
            ov.save_model(ov_model, ir_path)
            del ov_model
        cleanup_torchscript_cache()
        print("Text Encoder successfully converted to IR")


if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder(pipe.text_encoder, TEXT_ENCODER_OV_PATH)
    del pipe.text_encoder
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")


def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model to IR format.
    Function accepts pipeline, creates wrapper class for export only necessary for inference part,
    prepares example inputs for convert,
    Parameters:
        vae (torch.nn.Module): VAE model
        ir_path (Path): File for storing model
    Returns:
        None
    """

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not ir_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latent_sample = torch.randn((1, 4, 128, 128))

        vae_decoder.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(
                vae_decoder,
                example_input=latent_sample,
                input=[
                    (-1, 4, -1, -1),
                ],
            )
            
            ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print("VAE decoder successfully converted to IR")


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder(pipe.vae, VAE_DECODER_OV_PATH)
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")


def reshape(
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
        tokenizer_max_length: int = -1,
):
    if not CONTROLNET_STATIC_OV_PATH.exists():
        controlnet = core.read_model(CONTROLNET_OV_PATH)
        def reshape_controlnet(
                model: ov.runtime.Model,
                batch_size: int = -1,
                height: int = -1,
                width: int = -1,
                num_images_per_prompt: int = -1,
                tokenizer_max_length: int = -1,
            ):
                if batch_size == -1 or num_images_per_prompt == -1:
                    batch_size = -1
                else:
                    batch_size *= num_images_per_prompt
                    # The factor of 2 comes from the guidance scale > 1
                    if "timestep_cond" not in {inputs.get_node().get_friendly_name() for inputs in model.inputs}:
                        batch_size *= 2

                height_ = height // 8 if height > 0 else height
                width_ = width // 8 if width > 0 else width
                shapes = {}
                for inputs in model.inputs:
                    shapes[inputs] = inputs.get_partial_shape()
                    if inputs.get_node().get_friendly_name() == "timestep":
                        shapes[inputs] = shapes[inputs]
                    elif inputs.get_node().get_friendly_name() == "sample":
                        shapes[inputs] = [2, 4, height_, width_]
                    elif inputs.get_node().get_friendly_name() == "controlnet_cond":
                        shapes[inputs][0] = batch_size
                        shapes[inputs][2] = height 
                        shapes[inputs][3] = width  
                    elif inputs.get_node().get_friendly_name() == "time_ids":
                        shapes[inputs] = [batch_size, 6]
                    elif inputs.get_node().get_friendly_name() == "text_embeds":
                        shapes[inputs] = [batch_size, 1280]
                    elif inputs.get_node().get_friendly_name() == "encoder_hidden_states":
                        shapes[inputs][0] = batch_size
                        shapes[inputs][1] = tokenizer_max_length
                model.reshape(shapes)
                model.validate_nodes_and_infer_types()
                
        reshape_controlnet(controlnet, batch_size, height, width, num_images_per_prompt, tokenizer_max_length)
        ov.save_model(controlnet, CONTROLNET_STATIC_OV_PATH)

    if not UNET_STATIC_OV_PATH.exists():
        unet = core.read_model(UNET_OV_PATH)
        def reshape_unet_controlnet(
            model: ov.runtime.Model,
            batch_size: int = -1,
            height: int = -1,
            width: int = -1,
            num_images_per_prompt: int = -1,
            tokenizer_max_length: int = -1,
        ):
            if batch_size == -1 or num_images_per_prompt == -1:
                batch_size = -1
            else:
                batch_size *= num_images_per_prompt
                # The factor of 2 comes from the guidance scale > 1
                if "timestep_cond" not in {inputs.get_node().get_friendly_name() for inputs in model.inputs}:
                    batch_size *= 2

            height = height // 8 if height > 0 else height
            width = width // 8 if width > 0 else width
            shapes = {}
            for inputs in model.inputs:
                shapes[inputs] = inputs.get_partial_shape()
                if inputs.get_node().get_friendly_name() == "timestep":
                    shapes[inputs] = shapes[inputs]
                elif inputs.get_node().get_friendly_name() == "sample":
                    shapes[inputs] = [2, 4, height, width]
                elif inputs.get_node().get_friendly_name() == "text_embeds":
                    shapes[inputs] = [batch_size, 1280]
                elif inputs.get_node().get_friendly_name() == "time_ids":
                    shapes[inputs] = [batch_size, 6]
                elif inputs.get_node().get_friendly_name() == "encoder_hidden_states":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][1] = tokenizer_max_length
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_1":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height 
                    shapes[inputs][3] = width    
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_3":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height  
                    shapes[inputs][3] = width      
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_5":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height   
                    shapes[inputs][3] = width     
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_7":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 2 
                    shapes[inputs][3] = width // 2  
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_9":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 2 
                    shapes[inputs][3] = width // 2      
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_11":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 2 
                    shapes[inputs][3] = width // 2    
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_13":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 4 
                    shapes[inputs][3] = width // 4    
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_15":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 4 
                    shapes[inputs][3] = width // 4    
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_17":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 4 
                    shapes[inputs][3] = width // 4  
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_19":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 8
                    shapes[inputs][3] = width // 8   
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_21":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 8
                    shapes[inputs][3] = width // 8   
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 8
                    shapes[inputs][3] = width // 8   
                elif inputs.get_node().get_friendly_name() == "mid_block_additional_residual":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 8 
                    shapes[inputs][3] = width // 8   

            model.reshape(shapes)
            model.validate_nodes_and_infer_types()

        reshape_unet_controlnet(unet, batch_size, height, width, num_images_per_prompt, tokenizer_max_length)
        ov.save_model(unet, UNET_STATIC_OV_PATH)

    if not TEXT_ENCODER_STATIC_OV_PATH.exists() :#or  not TEXT_ENCODER_STATIC_2_OV_PATH.exists():
        text_encoder = core.read_model(TEXT_ENCODER_OV_PATH)
        # text_encoder_2 = core.read_model(TEXT_ENCODER_2_OV_PATH)

        def reshape_text_encoder(
            model: ov.runtime.Model, batch_size: int = -1, tokenizer_max_length: int = -1
        ):
            if batch_size != -1:
                shapes = {model.inputs[0]: [batch_size, tokenizer_max_length]}
                model.reshape(shapes)
                model.validate_nodes_and_infer_types()

        reshape_text_encoder(text_encoder, 1, tokenizer_max_length)
        # reshape_text_encoder(text_encoder_2, 1, tokenizer_max_length)
        ov.save_model(text_encoder, TEXT_ENCODER_STATIC_OV_PATH)
        # ov.save_model(text_encoder_2, TEXT_ENCODER_STATIC_2_OV_PATH)

    if not VAE_DECODER_STATIC_OV_PATH.exists():
        vae_decoder = core.read_model(VAE_DECODER_OV_PATH)
        def reshape_vae_decoder(model: ov.runtime.Model, height: int = -1, width: int = -1):
            height = height // 8 if height > -1 else height
            width = width // 8 if width > -1 else width
            latent_channels = 4
            shapes = {model.inputs[0]: [1, latent_channels, height, width]}
            model.reshape(shapes)
            model.validate_nodes_and_infer_types()

        reshape_vae_decoder(vae_decoder, height, width)
        ov.save_model(vae_decoder, VAE_DECODER_STATIC_OV_PATH)


#convert to static model
NEED_STATIC = True
STATIC_SHAPE = [1024,1024]
DEVICE_NAME = "GPU.1"

if NEED_STATIC and not os.path.exists("ov_models_static"):
        print("Converting to static models")
        if os.path.exists("ov_models_dynamic"):
            reshape(
            batch_size=1,
            height=STATIC_SHAPE[0],
            width=STATIC_SHAPE[1],
            num_images_per_prompt=1,
            tokenizer_max_length=77,
            )   
        else:
            raise ValueError("No ov_models_dynamic exists, please try ov_model_export.py with CONVERT = True first")
        
#add lora to model
NEED_LORA = True
DEVICE_NAME = "GPU.1"
LORA_PATH = "lora/01_Commercial_Complex.safetensors"
if NEED_STATIC:
    MODEL_PATH = "ov_models_static"
    MODEL_LOAR_PATH = "ov_models_static_lora"
else:
    MODEL_PATH = "ov_models_dynamic"
    MODEL_LOAR_PATH = "ov_models_dynamic_lora"

if NEED_LORA :
    from util import convert_model_to_lora
    convert_model_to_lora(MODEL_PATH, MODEL_LOAR_PATH, LORA_PATH, DEVICE_NAME)
