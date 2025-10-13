from torch import nn

class DiffusionLora(nn.Module):
    def __init__(self, pretrained_model_name_or_path,  rank, lora_modules, init_lora_weights, weight_dtype, device, target_size=1024):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.target_size = target_size
        self.device = device
        self.lora_rank = rank
        self.lora_modules = lora_modules
        self.init_lora_weights = init_lora_weights
        
        # TO DO
        # self.weight_dtype = 
        # self.tokenizer = 
        # self.text_encoder =
        # self.noise_scheduler =
        # self.vae = 
        # self.unet =


        
    def prepare_for_training(self):
        self.vae.requires_grad_(False)
        # TO DO
        # activate\disactivate grad from modules
        # pass modules to dtype
        # initialize lora config
        


    def get_trainable_params(self, config):
        # return list of trainable parameters
        # trainable_params = [
        #     {'params': ..., 'lr': ..., 'name': ...},
        #     ....
        # ]
        return trainable_params

    def get_state_dict(self):
        # return state dict of the trainable model
        return ...

    def load_state_dict_(self, state_dict):
        pass
        # load state_dict to the model
        

    def forward(self, pixel_values, prompt, do_cfg=False, *args, **kwargs):
        # pixel_values -- torch.Tensor size of bs x 3 x H x W
        # prompt -- list of str
        # do_cfg -- bool, to perform classifier free guidance or not
        
        # TO DO
        # pass pixel_values to vae and noise obtained latents
        # encode prompt and gain text embeds
        # do forward of unet
                
        return {
            'model_pred': ...,
            'target': ...,
        }
