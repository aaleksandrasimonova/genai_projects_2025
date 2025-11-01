import torch
from diffusers import DiffusionPipeline

class SDXLPipeline:
    def from_pretrained(model, *args, **kwargs):
        # TO DO
        # create pipeline
        # initizlize it with model submodules (like model.unet, model.text_encoder and etc.)
        # pipeline = ...

        pipeline = DiffusionPipeline.from_pretrained(
            model.pretrained_model_name_or_path
        )
        
        pipeline.unet = model.unet
        pipeline.text_encoder = model.text_encoder
        pipeline.text_encoder_2 = model.text_encoder_2
        pipeline.vae = model.vae
        pipeline.tokenizer = model.tokenizer
        pipeline.tokenizer_2 = model.tokenizer_2
        pipeline.scheduler = model.noise_scheduler
        
        pipeline = pipeline.to(model.device)
        pipeline = pipeline.to(dtype=model.weight_dtype)
        
        return pipeline