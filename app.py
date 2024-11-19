#!/usr/bin/env python
# credits: <https://github.com/mit-han-lab/hart>

import time
import numpy as np
import torch
from rich import print as pprint
import gradio as gr


pipe = None


class HARTPipeline():
    from hart.modules.models.transformer import HARTForT2I # registers automodel handler

    model_path: str = "mit-han-lab/hart-0.7b-1024px"
    encoder_path: str = "mit-han-lab/Qwen2-VL-1.5B-Instruct"
    device = torch.device('cuda')
    dtype = torch.float16
    model: HARTForT2I = None
    encoder = None
    tokenizer = None

    def __init__(self,
                 model_path: str = None,
                 encoder_path: str = None,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                ):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or self.device
        self.dtype = dtype or self.dtype
        self.model_path = model_path or self.model_path
        self.encoder_path = encoder_path or self.encoder_path
        self.model = AutoModel.from_pretrained(
            self.model_path,
            subfolder='llm',
            torch_dtype=self.dtype,
            vae_path=self.model_path,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_path)
        self.encoder = AutoModel.from_pretrained(self.encoder_path, torch_dtype=self.dtype)
        self.encoder.eval()


    def encode(self, prompt: str|list[str], max_tokens: int = 300, llm: bool = True, batch: int = 1):
        from hart.utils import encode_prompts, llm_system_prompt
        prompts = batch * [prompt] if isinstance(prompt, str) else prompt
        pipe.encoder.to(self.device)
        (
            _context_tokens,
            context_mask,
            context_ids,
            context_tensor,
        ) = encode_prompts(
            prompts=prompts,
            text_model=pipe.encoder,
            text_tokenizer=pipe.tokenizer,
            text_tokenizer_max_length=max_tokens,
            system_prompt=llm_system_prompt,
            use_llm_system_prompt=llm,
        )
        pipe.encoder.to('cpu')
        return context_mask, context_ids, context_tensor

    def seed(self, seed: int):
        import random
        from transformers import set_seed
        if seed == -1:
            seed = random.randint(0, np.iinfo(np.int32).max)
        set_seed(seed)
        return seed

    def generate(self,
                 context_tensor,
                 context_ids,
                 context_mask,
                 guidance: float = 4.5,
                 smooth: bool = True,
                 iterations: int = 1,
                 top_k: int = 600,
                 top_p: float = 0,
                 seed: int = -1,
                ):
        seed = self.seed(seed)
        pipe.model.to(self.device)
        samples = pipe.model.autoregressive_infer_cfg(
            B=context_tensor.size(0), # batch size comes from number of prompts
            label_B=context_tensor,
            top_k=top_k,
            top_p=top_p,
            cfg=guidance,
            g_seed=seed,
            num_maskgit_iters=iterations,
            more_smooth=smooth,
            context_position_ids=context_ids,
            context_mask=context_mask,
        )
        pipe.model.to('cpu')
        return samples

    def sample(self, samples: torch.Tensor):
        from PIL import Image
        samples = (255 * samples.float().cpu().numpy()).clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
        images = [Image.fromarray(sample) for sample in samples]
        return images


def mem():
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    used = round((total - free) / 1024 / 1024)
    return used


def generate(
    prompt: str,
    seed: int = 0,
    batch: int = 1,
    guidance: float = 4.5,
    smooth: bool = True,
    llm: bool = True,
    iterations: int = 1,
    top_k: int = 600,
    top_p: float = 0,
    max_tokens: int = 300,
    progress=gr.Progress(track_tqdm=True),
):
    global pipe
    if pipe is None:
        pprint('loading...')
        t0 = time.time()
        pipe = HARTPipeline()
        t1 = time.time()
        pprint(f'load: model="{pipe.model_path}" cls={pipe.model.__class__.__name__} encoder="{pipe.encoder_path}" cls={pipe.encoder.__class__.__name__} device={pipe.device} dtype={pipe.dtype} memory={mem()} time={t1-t0:.3f}')
    pprint(f'request: seed={seed} batch={batch} guidance={guidance} smooth={smooth} llm={llm} iterations={iterations} top_k={top_k} top_p={top_p} max_tokens={max_tokens} prompt="{prompt}" ')
    with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=pipe.dtype, cache_enabled=False):
        t0 = time.time()
        context_mask, context_ids, context_tensor = pipe.encode(
            prompt,
            max_tokens,
            llm,
            batch,
        )
        t1 = time.time()
        pprint(f'encode: context={context_tensor.shape} memory={mem()} time={t1-t0:.3f}')
        samples = pipe.generate(
            context_tensor=context_tensor,
            context_ids=context_ids,
            context_mask=context_mask,
            guidance=guidance,
            smooth=smooth,
            iterations=iterations,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        t2 = time.time()
        images = pipe.sample(samples)
    pprint(f'generate: images={images} memory={mem()} time={t2-t1:.3f} ')
    return images


if __name__ == "__main__":
    mem()
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Column(scale=1):
                with gr.Row():
                    prompt = gr.Text(label="Prompt", show_label=False, max_lines=2, placeholder="prompt", container=False)
                    run_button = gr.Button("Run", scale=0)
                with gr.Row():
                    smooth = gr.Checkbox(label="Smoother output", value=True)
                    llm = gr.Checkbox(label="Prompt enhancer", value=True)
                with gr.Row():
                    seed = gr.Number(label="Seed", minimum=-1, maximum=np.iinfo(np.int32).max, step=1, value=-1)
                    top_k = gr.Number(label="Top-K", minimum=0, maximum=2000, step=1, value=600)
                    top_p = gr.Number(label="Top-P", minimum=0, maximum=1, step=0.01, value=0)
                    batch = gr.Slider(label="Batch size", minimum=1, maximum=16, step=1, value=1)
                    guidance = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=20, step=0.1, value=4.5)
                    iterations = gr.Slider(label="Iterations", minimum=1, maximum=99, step=1, value=1)
            with gr.Column(scale=4):
                with gr.Row():
                    result = gr.Gallery(label="Result", columns=2, show_label=False)
        gr.on(
            triggers=[prompt.submit, run_button.click],
            fn=generate,
            inputs=[prompt, seed, batch, guidance, smooth, llm, iterations, top_k, top_p], 
            outputs=[result],
            api_name="run",
        )
    demo.queue(max_size=20).launch()
