
import numpy as np
from pathlib import Path
import PIL
import torch
from diffusers.utils import  numpy_to_pil, load_image
from util import load_lora_runtime
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers import ConfigMixin, PNDMScheduler
from transformers import  CLIPTokenizer
import openvino as ov
from tqdm.auto import tqdm
from optimum.pipelines.diffusers.pipeline_utils import VaeImageProcessor

class StableDiffusionContrlNetPipelineMixin(ConfigMixin):
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
        controlnet_conditioning_scale: float = 1.0,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        if self.has_lora:
            text_input = self.lora_text_encoder_input_value_dict
            text_input['input_ids'] = text_input_ids
        else:
            text_input = text_input_ids

        text_embeddings = self.text_encoder(text_input)[0]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            uncond_text_input_ids = uncond_input.input_ids
            if self.has_lora:
                uncond_text_input = self.lora_text_encoder_input_value_dict
                uncond_text_input['input_ids'] = uncond_text_input_ids
            else:
                uncond_text_input = uncond_text_input_ids
            uncond_embeddings = self.text_encoder(uncond_text_input)[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: np.dtype = np.float32,
        latents: np.ndarray = None,
    ):
        """
        Preparing noise to image generation. If initial latents are not provided, they will be generated randomly,
        then prepared latents scaled by the standard deviation required by the scheduler

        Parameters:
           batch_size (int): input batch size
           num_channels_latents (int): number of channels for noise generation
           height (int): image height
           width (int): image width
           dtype (np.dtype, *optional*, np.float32): dtype for latents generation
           latents (np.ndarray, *optional*, None): initial latent noise tensor, if not provided will be generated
        Returns:
           latents (np.ndarray): scaled initial noise for diffusion
        """
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = self.randn_tensor(shape, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: np.array, pad: Tuple[int]):
        """
        Decode predicted image from latent space using VAE Decoder and unpad image result

        Parameters:
           latents (np.ndarray): image encoded in diffusion latent space
           pad (Tuple[int]): each side padding sizes obtained on preprocessing step
        Returns:
           image: decoded by VAE decoder image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[0]
        (_, end_h), (_, end_w) = pad[1:3]
        h, w = image.shape[2:]
        unpad_h = h - end_h
        unpad_w = w - end_w
        image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        return image

    def scale_fit_to_window(self, dst_width: int, dst_height: int, image_width: int, image_height: int):
        """
        Preprocessing helper function for calculating image size for resize with peserving original aspect ratio
        and fitting image to specific window size

        Parameters:
        dst_width (int): destination window width
        dst_height (int): destination window height
        image_width (int): source image width
        image_height (int): source image height
        Returns:
        result_width (int): calculated width for resize
        result_height (int): calculated height for resize
        """
        im_scale = min(dst_height / image_height, dst_width / image_width)
        return int(im_scale * image_width), int(im_scale * image_height)

    def preprocess(self, image: PIL.Image.Image, height, width):
        """
        Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
        then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
        converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
        The function returns preprocessed input tensor and padding size, which can be used in postprocessing.

        Parameters:
        image (PIL.Image.Image): input image
        Returns:
        image (np.ndarray): preprocessed image tensor
        pad (Tuple[int]): pading size for each dimension for restoring image size in postprocessing
        """
        src_width, src_height = image.size
        dst_width, dst_height = self.scale_fit_to_window(width, height, src_width, src_height)
        image = np.array(image.resize((dst_width, dst_height), resample=PIL.Image.Resampling.LANCZOS))[None, :]
        pad_width = width - dst_width
        pad_height = height - dst_height
        pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
        image = np.pad(image, pad, mode="constant")
        image = image.astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        return image, pad

    def randn_tensor(
        self,
        shape: Union[Tuple, List],
        dtype: Optional[np.dtype] = np.float32,
    ):
        """
        Helper function for generation random values tensor with given shape and data type

        Parameters:
        shape (Union[Tuple, List]): shape for filling random values
        dtype (np.dtype, *optiona*, np.float32): data type for result
        Returns:
        latents (np.ndarray): tensor with random values with given data type and shape (usually represents noise in latent space)
        """
        latents = np.random.randn(*shape).astype(dtype)

        return latents

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PIL.Image.Image,
        num_inference_steps: int = 10,
        negative_prompt: Union[str, List[str]] = None,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        eta: float = 0.0,
        latents: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Parameters:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`PIL.Image.Image`):
                `PIL.Image`, or tensor representing an image batch which will be repainted according to `prompt`.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            negative_prompt (`str` or `List[str]`):
                negative prompt or prompts for generation
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. This pipeline requires a value of at least `1`.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
        Returns:
            image ([List[Union[np.ndarray, PIL.Image.Image]]): generaited images

        """

        # 1. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # 2. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, negative_prompt=negative_prompt)

        # 3. Preprocess image
        orig_width, orig_height = image.size
        image, pad = self.preprocess(image, height=height, width=width)
        height, width = image.shape[-2:]
        if do_classifier_free_guidance:
            image = np.concatenate(([image] * 2))

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.set_progress_bar_config(disable=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                control_model_input = {
                    "sample": latent_model_input,
                    "timestep": t,
                    "encoder_hidden_states": text_embeddings,
                    "controlnet_cond": image,
                }
                result = self.controlnet(control_model_input)
                
                # predict the noise residual
                unet_input = {
                    "sample": latent_model_input,
                    "timestep": t,
                    "encoder_hidden_states": text_embeddings,
                }
                layer_num = 0
                for key, value in result.items():
                    if layer_num == len(result) - 2:
                        name = "down_block_additional_residual"
                        layer_num += 1
                    elif layer_num == len(result) - 1:
                        name = "mid_block_additional_residual"
                        layer_num += 1
                    else:
                        name = "down_block_additional_residual." + str(layer_num*2 + 1)
                        layer_num += 1
                    unet_input[name] = controlnet_conditioning_scale * value

                # noise_pred = self.unet(unet_input)[0]
                if self.has_lora:
                    noise_pred = self.unet({**unet_input, **self.lora_unet_input_value_dict})[0]
                else:
                    noise_pred = self.unet({**unet_input})[0]  
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    torch.from_numpy(noise_pred), t, torch.from_numpy(latents)
                ).prev_sample.numpy()

                # update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        image = self.decode_latents(latents, pad)

        # 9. Convert to PIL
        image = numpy_to_pil(image)
        image = [img.resize((orig_width, orig_height), PIL.Image.Resampling.LANCZOS) for img in image]

        return image

class OVStableDiffusionControlNetPipeline(StableDiffusionContrlNetPipelineMixin):
    """
    OpenVINO inference pipeline for Stable Diffusion XL with ControlNet guidence
    """
    def __init__(
        self,
        scheduler,
        unet: ov.Model,
        controlnet: ov.Model,
        tokenizer: CLIPTokenizer,   
        text_encoder: ov.Model,
        # text_encoder_2: Optional[ov.Model],
        vae_decoder: ov.Model,
        device: str = "AUTO",
        lora_weights: Optional[list] = None,
    ):
        if lora_weights:
            self.lora_text_encoder_input_value_dict = lora_weights[0]
            # self.lora_text_encoder_2_input_value_dict = lora_weights[1]
            self.lora_unet_input_value_dict = lora_weights[2]
            self.has_lora = True
        else:
            self.has_lora = False
        print('self.has_lora: ', self.has_lora)
        self.text_encoder = text_encoder
        # self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        # self.tokenizer_2 = tokenizer_2
        self.controlnet = controlnet
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.scheduler = scheduler
        self.vae_scale_factor = 8
        self.vae_scaling_factor = 0.13025
        # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        # self.control_image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor,
        #     do_convert_rgb=True,
        #     do_normalize=False,
        # )
        # self._internal_dict = {}
        # self._progress_bar_config = {}

    def switch_lora(self , lora_weights: list):
        self.lora_text_encoder_input_value_dict = lora_weights[0]
        # self.lora_text_encoder_2_input_value_dict = lora_weights[1]
        self.lora_unet_input_value_dict = lora_weights[2]

    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[PIL.Image.Image] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        eta: float = 0.0,
        latents: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        return StableDiffusionContrlNetPipelineMixin.__call__(
            self,
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            eta=eta,
            latents=latents,
            height=height,
            width=width,
        )

DEVICE_NAME="GPU.1"

COMPILE_CONFIG_FP32 = {'INFERENCE_PRECISION_HINT': 'f32'}
COMPILE_CONFIG_FP16 = {'INFERENCE_PRECISION_HINT': 'f16'}

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

USE_LORA = True
UNET_LORA_OV_PATH = Path("./ov_models_static_lora/unet/openvino_model.xml")
TEXT_ENCODER_LORA_OV_PATH = Path("./ov_models_static_lora/text_encoder/openvino_model.xml")
LORA_PATH_01 = "lora/01_Commercial_Complex.safetensors"
LORA_PATH_02 = "lora/02_office_tower.safetensors"
LORA_PATH_03 = "lora/03_villa.safetensors"

core = ov.Core()

if not USE_LORA:
    import time
    start_time=time.time()
    controlnet = core.compile_model(CONTROLNET_STATIC_OV_PATH,device_name=DEVICE_NAME, config=COMPILE_CONFIG_FP16)
    unet = core.compile_model(UNET_STATIC_OV_PATH,device_name=DEVICE_NAME, config=COMPILE_CONFIG_FP16)
    text_encoder = core.compile_model(TEXT_ENCODER_STATIC_OV_PATH,device_name=DEVICE_NAME, config=COMPILE_CONFIG_FP16)
    vae_decoder = core.compile_model(VAE_DECODER_STATIC_OV_PATH,device_name=DEVICE_NAME, config=COMPILE_CONFIG_FP16)
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_OV_PATH)
    scheduler = PNDMScheduler.from_config(SCHEDULER_OV_PATH)

    ov_pipe = OVStableDiffusionControlNetPipeline(
        text_encoder=text_encoder,
        # text_encoder_2=text_encoder_2,
        controlnet=controlnet,
        unet=unet,
        vae_decoder=vae_decoder,
        tokenizer=tokenizer,
        # tokenizer_2=tokenizer_2,
        scheduler=scheduler,
    )
    end_time=time.time()
    print("pipeline init cost time(s): ")
    print(end_time-start_time)

    seed = 42
    torch.manual_seed(seed)           
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    controlnet_conditioning_scale = 1

    prompt = "commercial_complex, A rendering of the exterior front facade, an office building with glass,"
    negative_prompt = "(blue long upper shan:1.3),(lightcyan:1.3),dark,blurry,unappealing,noisy,unprofessional,over sharpening,dirt,bad color matching,graying,"

    image = load_image("./line.png")


    start_time=time.time()

    images = ov_pipe(
        prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=20,height=1024,width=1024,
        )
    end_time=time.time()
    print("infer cost time(s): ")
    print(end_time-start_time)
    images[0].save(f"result.png") 
else:
    import time
    start_time=time.time()
    loras = load_lora_runtime(LORA_PATH_01, DEVICE_NAME), load_lora_runtime(LORA_PATH_02, DEVICE_NAME), load_lora_runtime(LORA_PATH_03, DEVICE_NAME)
    end_time=time.time()
    print("load lora cost time(s): ")
    print(end_time-start_time)

    start_time=time.time()
    controlnet = core.compile_model(CONTROLNET_STATIC_OV_PATH,device_name=DEVICE_NAME, config=COMPILE_CONFIG_FP16)
    unet = core.compile_model(UNET_LORA_OV_PATH,device_name=DEVICE_NAME, config=COMPILE_CONFIG_FP16)
    text_encoder = core.compile_model(TEXT_ENCODER_LORA_OV_PATH,device_name=DEVICE_NAME, config=COMPILE_CONFIG_FP16)
    vae_decoder = core.compile_model(VAE_DECODER_STATIC_OV_PATH,device_name=DEVICE_NAME, config=COMPILE_CONFIG_FP16)
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_OV_PATH)
    # tokenizer_2 = CLIPTokenizer.from_pretrained(TOKENIZER_2_OV_PATH)
    scheduler = PNDMScheduler.from_pretrained(SCHEDULER_OV_PATH)

    print('compiled success')
    ov_pipe = OVStableDiffusionControlNetPipeline(
        text_encoder=text_encoder,
        # text_encoder_2=text_encoder_2,
        controlnet=controlnet,
        unet=unet,
        vae_decoder=vae_decoder,
        tokenizer=tokenizer,
        # tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        lora_weights = loras[0]
    )
    end_time=time.time()
    print("pipeline init cost time(s): ")
    print(end_time-start_time)

    seed = 42
    torch.manual_seed(seed)           
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    controlnet_conditioning_scale = 1.0

    prompt = "commercial_complex, A rendering of the exterior front facade, an office building with glass,"
    negative_prompt = "(blue long upper shan:1.3),(lightcyan:1.3),dark,blurry,unappealing,noisy,unprofessional,over sharpening,dirt,bad color matching,graying,"

    image = load_image("./line.png")

    start_time=time.time()

    images = ov_pipe(
        prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=20,height=1024,width=1024,
        )
    end_time=time.time()
    print("infer cost time(s): ")
    print(end_time-start_time)
    images[0].save(f"result_lora_01.png") 

    controlnet_conditioning_scale = 0.5
    ov_pipe.switch_lora(loras[1])
    prompt = "office_tower, good weather, skyline, A rendering of the exterior front facade"
    start_time=time.time()
    images = ov_pipe(
        prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=20,height=1024,width=1024,
        )
    end_time=time.time()
    print("infer cost time(s): ")
    print(end_time-start_time)
    images[0].save(f"result_lora_02.png") 

    controlnet_conditioning_scale = 1.0
    ov_pipe.switch_lora(loras[2])
    prompt = "villa, A rendering of the exterior front facade, good weather"
    start_time=time.time()
    images = ov_pipe(
        prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=20,height=1024,width=1024,
        )
    end_time=time.time()
    print("infer cost time(s): ")
    print(end_time-start_time)
    images[0].save(f"result_lora_03.png") 