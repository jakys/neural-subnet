import torch
from .utils import seed_everything, timing_decorator, auto_amp_inference
from .utils import get_parameter_number, set_parameter_grad_false
from diffusers import HunyuanDiTPipeline, AutoPipelineForText2Image,SanaPipeline

class SNText2Image():
    def __init__(self, pretrain="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", device="cuda:0", save_memory=False):
        '''
            save_memory: if GPU memory is low, can set it
        '''
        self.save_memory = save_memory
        self.device = device
        self.pipe = SanaPipeline.from_pretrained(
            pretrain,
            torch_dtype=torch.float16,
            device_map="balanced",
        )
        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.transformer = self.pipe.transformer.to(torch.bfloat16)

    @torch.no_grad()
    @timing_decorator('sana text to image')
    @auto_amp_inference
    def __call__(self, *args, **kwargs):
        if self.save_memory:
            self.pipe = self.pipe.to(self.device)
            torch.cuda.empty_cache()
            res = self.call(*args, **kwargs)
            self.pipe = self.pipe.to("cpu")
        else:
            res = self.call(*args, **kwargs)
        torch.cuda.empty_cache()
        return res

    def call(self, prompt, seed=42, steps=20):
        '''
            inputs:
                prompr: str
                seed: int
                steps: int
            return:
                rgb: PIL.Image
        '''
        prompt = prompt + ' white background cartoon clarity 3D stereo'
        seed_everything(seed)
        rgb = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=steps,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            return_dict=False
        )[0][0]
        torch.cuda.empty_cache()
        return rgb
    