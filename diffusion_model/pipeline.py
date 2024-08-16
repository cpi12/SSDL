import torch
import numpy as np
from tqdm import tqdm
from diffusion_model import util
from diffusion_model import model_loader

def generate(
        diffusion_model,
        real_data_loader,
        batch_size,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler="k_lms",
        n_inference_steps=5,
        models={},
        seed=None,
        device=None,
        idle_device=None
):
    with torch.no_grad():
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        sensor_model = models.get('sensor_model') or model_loader.load_sensor_model(device)
        sensor_model.to(device)

        all_skeletons = []

        for real_data in real_data_loader:
            real_skeleton, real_sensor1, real_sensor2, _ = real_data

            real_skeleton = real_skeleton.to(device)
            real_sensor1 = real_sensor1.to(device)
            real_sensor2 = real_sensor2.to(device)

            cond_context = sensor_model(real_sensor1, real_sensor2, return_attn_output=True)
            context = cond_context if do_cfg else None

            noise_shape = (real_skeleton.size(0), 90, 96)  # Adjust the noise shape for skeletons

            latents = torch.randn(noise_shape, generator=generator, device=device)
            latents *= strength

            timesteps = tqdm(range(n_inference_steps))
            for i, timestep in enumerate(tqdm(timesteps, desc="Inference")):
                time_embedding = util.get_time_embedding(timestep, torch.float32).to(device)
                noise = torch.randn(noise_shape, generator=generator, device=device)

                input_latents = latents
                if do_cfg:
                    input_latents = input_latents.repeat(2, 1, 1)
                # print(time_embedding.shape)
                output = diffusion_model(input_latents, context, torch.rand(input_latents.size(1), device=device), noise)
                if do_cfg:
                    output_cond, output_uncond = output.chunk(2)
                    output = cfg_scale * (output_cond - output_uncond) + output_uncond

                latents = latents - output

            all_skeletons.append(latents)

        all_skeletons = torch.cat(all_skeletons, dim=0)
        decoder = models.get('decoder') or model_loader.load_decoder(device)
        decoder.to(device)
        skeletons = decoder(all_skeletons)
        to_idle(decoder)
        del decoder

        skeletons = skeletons.to('cpu', torch.float32).numpy()

        return [torch.tensor(skeleton) for skeleton in skeletons]
