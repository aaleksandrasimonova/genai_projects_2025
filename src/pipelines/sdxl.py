import torch
from diffusers import DiffusionPipeline

class SDXLPipeline:
    def from_pretrained(model, *args, **kwargs):
        # TO DO
        # create pipeline
        # initizlize it with model submodules (like model.unet, model.text_encoder and etc.)
        # pipeline = ...

        pipeline = DiffusionPipeline.from_pretrained(
            model.pretrained_model_name_or_path,
            unet=model.unet,
            text_encoder=model.text_encoder,
            text_encoder_2=model.text_encoder_2,
            vae=model.vae,
            tokenizer=model.tokenizer,
            tokenizer_2=model.tokenizer_2,
            scheduler=model.noise_scheduler,
            torch_dtype=model.weight_dtype,
            *args,
            **kwargs
        )
        
        pipeline = pipeline.to(model.device)
        
        return pipeline 