import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

class DiffusionLora(nn.Module):
    def __init__(self, pretrained_model_name_or_path,  rank, lora_modules, init_lora_weights, weight_dtype, device, target_size=1024):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.target_size = target_size
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.lora_rank = rank
        self.lora_modules = lora_modules
        self.init_lora_weights = init_lora_weights
        
        # TO DO
        if isinstance(weight_dtype, str):
            wd_lower = weight_dtype.lower()
            if wd_lower in ("fp16", "float16"):
                self.weight_dtype = torch.float16
            elif wd_lower in ("bf16", "bfloat16"):
                self.weight_dtype = torch.bfloat16
            elif wd_lower in ("float32", "fp32"):
                self.weight_dtype = torch.float32
        else:
            self.weight_dtype = weight_dtype

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="text_encoder_2",
            torch_dtype=self.weight_dtype
        )
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler"
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=torch.float32
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype
        )

        self.vae.to(self.device, dtype=torch.float32)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(self.device, dtype=self.weight_dtype)
        self.unet.to(self.device, dtype=self.weight_dtype)        

        
    def prepare_for_training(self):
        self.vae.requires_grad_(False)
        # TO DO
        # activate\disactivate grad from modules
        # pass modules to dtype
        # initialize lora config
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            target_modules=self.lora_modules,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=self.init_lora_weights,
        )
        self.unet = get_peft_model(self.unet, lora_config)

        for n, p in self.unet.named_parameters():
            if "lora" in n.lower() or "adapter" in n.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False


    def get_trainable_params(self, config):
        # return list of trainable parameters
        # trainable_params = [
        #     {'params': ..., 'lr': ..., 'name': ...},
        #     ....
        # ]
        trainable_params = [
            {
                "params": [p for p in self.unet.parameters() if p.requires_grad],
                "lr": getattr(config, "lr", 1e-2),
                "name": "unet_lora",
            }
        ]
        return trainable_params

    def get_state_dict(self):
        # return state dict of the trainable model
        try:
            return get_peft_model_state_dict(self.unet)
        except Exception:
            return self.unet.state_dict()

    def load_state_dict_(self, state_dict):
        # load state_dict to the model
        try:
            set_peft_model_state_dict(self.unet, state_dict)
        except Exception:
            self.unet.load_state_dict(state_dict, strict=False)

    
    def _encode_prompt(self, prompt, do_cfg=False):
        batch_size = len(prompt) if isinstance(prompt, (list, tuple)) else 1

        text_enc1 = self.tokenizer(prompt,
                              padding="max_length",
                              max_length=self.tokenizer.model_max_length,
                              truncation=True,
                              return_tensors="pt")

        text_enc2 = self.tokenizer_2(prompt,
                                padding="max_length",
                                max_length=self.tokenizer_2.model_max_length,
                                truncation=True,
                                return_tensors="pt")

        with torch.no_grad():
            out1 = self.text_encoder(text_enc1.input_ids.to(self.device),
                                     output_hidden_states=True)
            hidden1 = out1.hidden_states[-2]
            # pooled1 = out1.last_hidden_state[:, -1, :]

            out2 = self.text_encoder_2(text_enc2.input_ids.to(self.device),
                                       output_hidden_states=True)
            # hidden2 = out2.hidden_states[-2]
            pooled2 = out2.text_embeds #out2.last_hidden_state[:, -1, :] 

        # prompt_embeds = torch.cat([hidden1, hidden2], dim=-1)
        # pooled_prompt_embeds = torch.cat([pooled1, pooled2], dim=-1)

        prompt_embeds = hidden1
        pooled_prompt_embeds = pooled2

        if do_cfg:
            tokens_unc = [""] * batch_size
            text_enc1_unc = self.tokenizer(tokens_unc,
                                           padding="max_length",
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True,
                                           return_tensors="pt")
            
            text_enc2_unc = self.tokenizer_2(tokens_unc,
                                             padding="max_length",
                                             max_length=self.tokenizer_2.model_max_length,
                                             truncation=True,
                                             return_tensors="pt")

            with torch.no_grad():
                out1_unc = self.text_encoder(text_enc1_unc.input_ids.to(self.device),
                                           output_hidden_states=True)
                hidden1_unc = out1_unc.hidden_states[-2]
                # pooled1_unc = out1_unc.last_hidden_state[:, -1, :]

                out2_unc = self.text_encoder_2(text_enc2_unc.input_ids.to(self.device),
                                             output_hidden_states=True)
                # hidden2_unc = out2_unc.hidden_states[-2]
                pooled2_unc = out2_unc.text_embeds #out2_unc.last_hidden_state[:, -1, :]

            # prompt_embeds_unc = torch.cat([hidden1_unc, hidden2_unc], dim=-1)
            # pooled_prompt_embeds_unc = torch.cat([pooled1_unc, pooled2_unc], dim=-1)

            # prompt_embeds = torch.cat([prompt_embeds_unc, prompt_embeds], dim=0)
            # pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_unc, pooled_prompt_embeds], dim=0)

            prompt_embeds = torch.cat([hidden1_unc, hidden1], dim=0)
            pooled_prompt_embeds = torch.cat([pooled2_unc, pooled2], dim=0)

        return prompt_embeds, pooled_prompt_embeds
        

    def forward(self, pixel_values, prompt, do_cfg=False, cfg_scale=7.5, *args, **kwargs):
        # pixel_values -- torch.Tensor size of bs x 3 x H x W
        # prompt -- list of str
        # do_cfg -- bool, to perform classifier free guidance or not
        
        # TO DO
        # pass pixel_values to vae and noise obtained latents
        # encode prompt and gain text embeds
        # do forward of unet

        batch_size = pixel_values.shape[0]

        if pixel_values.shape[-2:] != (self.target_size, self.target_size):
            pixel_values = F.interpolate(
                pixel_values, 
                size=(self.target_size, self.target_size),
                mode='bilinear', 
                align_corners=False
            )

        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(device=self.device, dtype=self.weight_dtype)

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt, do_cfg)

        original_size = (self.target_size, self.target_size)
        target_size = (self.target_size, self.target_size)
        add_time_ids = torch.tensor(
            [list(original_size) + [0, 0] + list(target_size)],
            device=self.device,
            dtype=self.weight_dtype
            )

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids.repeat(pooled_prompt_embeds.shape[0], 1),
        }

        if do_cfg:
            timesteps = torch.cat([timesteps] * 2)
            noisy_latents = torch.cat([noisy_latents] * 2)

        model_pred = self.unet(
            noisy_latents,
            timesteps, 
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        if do_cfg:
            model_pred_uncond, model_pred_text = model_pred.chunk(2)
            model_pred = model_pred_uncond + cfg_scale * (model_pred_text - model_pred_uncond)


        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            # note: timesteps must match original latents shape for get_velocity; if do_cfg then take first half
            if do_cfg:
                half = latents.shape[0] // 2
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps[:half])
            else:
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)

        return {"model_pred": model_pred, "target": target}