"""Text edition with SD3 model on CreativeBench."""

import logging
import os
import time

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_msssim import ssim
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from diffusers.training_utils import set_seed
from src.eval.basic_metrics import calculate_mse, calculate_psnr_from_mse
from src.eval.clipscore import clip_metrics, extract_all_images
from src.eval.ocr_eval import get_ocr_easyocr, get_text_easyocr, ocr_metrics
from src.eval.text_distance import get_levenshtein_distances
from src.prepare_custom_dataset import prepare_toxic_bench, prepare_toxic_bench_with_args, prepare_full_toxic_bench_with_args
# from src.eval.text_detection import setup_text_detection_model, remove_text_boxes

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
SDXL_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-3-medium-diffusers"
SEED = 42
N_SAMPLES_PER_PROMPT = 1
BATCH_SIZE = 8
DEVICE = "cuda"
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 7.0
TIMESTEP_START_PATCHING = 2
ATTENTIONS_TO_PATCH = [
    10,
]



def set_to_string(int_set):
    return "_A".join(str(num) for num in int_set)


SAVE_DIR = (
    f"/mfsnic/u/tblanchard/results_sd3/glyph_creative/edit/"
    f"{START_TIME}_"
    f"seed_{SEED}_"
    f"n_samples_per_prompt_{N_SAMPLES_PER_PROMPT}_"
    f"n_inference_steps_{NUM_INFERENCE_STEPS}_"
    f"guidance_scale_{GUIDANCE_SCALE}_"
    f"timestep_start_patching_{TIMESTEP_START_PATCHING}_"
    f"attentions_to_patch_{set_to_string(ATTENTIONS_TO_PATCH)}"
)

os.makedirs(SAVE_DIR, exist_ok=True)

logging.info(f"Seed: {SEED}")
logging.info(f"Device: {DEVICE}")
logging.info(f"Num inference steps: {NUM_INFERENCE_STEPS}")
logging.info(f"Batch size: {BATCH_SIZE}")
logging.info(f"Save dir: {SAVE_DIR}")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].shape[1], imgs[0].shape[0]
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        img = img.astype(np.uint8)
        grid.paste(Image.fromarray(img), box=(i % cols * w, i // cols * h))
    return grid


set_seed(SEED)

pipe = StableDiffusion3Pipeline.from_pretrained(
    SDXL_MODEL_NAME_OR_PATH,
    variant="fp16",
    use_safetensors=True,
    token=os.environ.get("HF_TOKEN"),
    torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
).to(DEVICE)

# NOTE : we want the progress bar to trigger the callbacks
# pipe.set_progress_bar_config(disable=True)

generator = torch.Generator().manual_seed(SEED)

# NOTE : either prepare toxic bench with unrelated non toxic words added or just toxic words themselves
# prompts_A, prompts_B = prepare_toxic_bench(
#     n_samples_per_prompt=N_SAMPLES_PER_PROMPT
# )

# Full toxic bench
prompts_A, prompts_B, prompts_C = prepare_full_toxic_bench_with_args(
    toxic_path="./src/data/train/toxic_words.txt",
    non_toxic_path="./src/data/train/toxic_mappings.txt",
    unrelated_non_toxic_path="./src/data/train/train_non_tox.txt",
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT
)
print("prompts_A", prompts_A)
print("prompts_B", prompts_B)
print("prompts_C", prompts_C)

logging.info(f"Number of prompts: {len(prompts_A)}")

noises = torch.randn(
    (N_SAMPLES_PER_PROMPT, 16, 128, 128),
    generator=generator,
    dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
)
noises = noises.repeat(len(prompts_A) // N_SAMPLES_PER_PROMPT, 1, 1, 1)

ocr_model = get_ocr_easyocr(use_cuda=(DEVICE == "cuda"))

clip_model, transform = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model.eval()

# text_detection_model = setup_text_detection_model(
#     "resources/DB_IC15_resnet50.onnx"
# )


def sample(
    prompts,
    noise,
    batch_size,
    num_inference_steps,
    generator,
    device,
    run_with_cache,
    attn_idx_to_patch=None,
    attn_heads_idx_to_patch=None,
    timestep_start_patching=0,
):  
    last_images = np.zeros((len(prompts), 1024, 1024, 3), dtype=np.uint8)
    # all_images = np.zeros((len(prompts)*(NUM_INFERENCE_STEPS-15), 1024, 1024, 3), dtype=np.uint8)

    # batch_num is coming from a modification of pipeline_stable_diffusion_3.py script directly
    # def latents_callback(pipeline, step, _, batch_num, callbacks_kwargs):
    #     if step >= 15 :
    #         latents = callbacks_kwargs["latents"].clone().detach()
    #         latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    #         print(latents.shape)
    #         image = pipeline.vae.decode(latents, return_dict=False)[0]
    #         image = (image / 2 + 0.5).clamp(0, 1)
    #         image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    #         image = image * 255

    #         # We should have different images here
    #         all_images[batch_num + step : batch_num + step + batch_size] = image.astype(np.uint8)

    #     return callbacks_kwargs


    with tqdm(total=len(prompts)) as pbar:
        for batch_num, batch_start in enumerate(range(0, len(prompts), batch_size)):
            prompt = prompts[batch_start : batch_start + batch_size]
            latent = noise[batch_start : batch_start + batch_size].to(device)
            images = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                latents=latent,
                run_with_cache=run_with_cache,
                attn_idx_to_patch=attn_idx_to_patch,
                output_type="np",
                batch_num=batch_num,
                attn_heads_idx_to_patch=attn_heads_idx_to_patch,
                timestep_start_patching=timestep_start_patching,
                guidance_scale=GUIDANCE_SCALE,

                # NOTE : for dataset generation don't need that we just need the last step
                # callback_on_step_end = latents_callback,
                # callback_on_step_end_tensor_inputs=["latents"],
            ).images
            # all_images[(batch_start + NUM_INFERENCE_STEPS) : (batch_start + NUM_INFERENCE_STEPS + batch_size)] = images.astype(np.uint8)
            last_images[batch_start: (batch_start + batch_size)] = (images*255).astype(np.uint8)
            pbar.update(len(prompt))
    
    return last_images, last_images

    # return all_images, last_images


def calculate_metrics(
    original_images_A,
    original_images_A_feats,
    images,
    texts_A,
    texts_B,
    prompts_A,
    prompts_B,
    device,
    batch_size,
):
    # calculate metrics per sample
    # 1. MSE
    mse = calculate_mse(original_images_A, images)
    # 2.PSNR
    psnr = calculate_psnr_from_mse(mse)
    # 3. SSIM
    ssim_val = ssim(
        torch.from_numpy(original_images_A.astype(np.float32)).permute((0, 3, 1, 2)),
        torch.from_numpy(images.astype(np.float32)).permute((0, 3, 1, 2)),
        data_range=255,
        size_average=False,
    ).numpy()
    # 4. OCR Acc/Prec/Rec
    ocr_texts = [
        get_text_easyocr(ocr_model, images[i]).lower() for i in range(images.shape[0])
    ]
    ocr_pr_A, ocr_rec_A, ocr_acc_A = ocr_metrics(ocr_texts, texts_A)
    ocr_pr_B, ocr_rec_B, ocr_acc_B = ocr_metrics(ocr_texts, texts_B)
    # 5. CLIPScore
    image_sim, prompt_A_sim, prompt_B_sim = clip_metrics(
        clip_model,
        images,
        original_images_A_feats,
        device,
        batch_size,
        prompts_A,
        prompts_B,
    )
    # 6. Levenshtein distance
    leve_A = get_levenshtein_distances(ocr_texts, texts_A)
    leve_B = get_levenshtein_distances(ocr_texts, texts_B)
    # calculate visual metrics with removed text
    # images_no_text = remove_text_boxes(images, text_detection_model)
    # original_images_A_no_text = remove_text_boxes(
    #     original_images_A, text_detection_model
    # )
    # MSE
    # mse_no_text = calculate_mse(original_images_A_no_text, images_no_text)
    # PSNR
    #psnr_no_text = calculate_psnr_from_mse(mse_no_text)
    # SSIM
    # ssim_val_no_text = ssim(
    #     torch.from_numpy(original_images_A_no_text.astype(np.float32)).permute(
    #         (0, 3, 1, 2)
    #     ),
    #     torch.from_numpy(images_no_text.astype(np.float32)).permute((0, 3, 1, 2)),
    #     data_range=255,
    #     size_average=False,
    # ).numpy()

    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim_val,
        "OCR_A_Prec": ocr_pr_A,
        "OCR_A_Rec": ocr_rec_A,
        "OCR_A_Acc": ocr_acc_A,
        "OCR_B_Prec": ocr_pr_B,
        "OCR_B_Rec": ocr_rec_B,
        "OCR_B_Acc": ocr_acc_B,
        "CLIPScore_image": image_sim,
        "CLIPScore_prompt_A": prompt_A_sim,
        "CLIPScore_prompt_B": prompt_B_sim,
        "Levenshtein_A": leve_A,
        "Levenshtein_B": leve_B,
        "Prompts_A": prompts_A,
        "Prompts_B": prompts_B,
        "OCR_texts": ocr_texts,
        "Texts_A": texts_A,
        "Texts_B": texts_B,
        # "MSE_no_text": mse_no_text,
        # "PSNR_no_text": psnr_no_text,
        # "SSIM_no_text": ssim_val_no_text,
    }


patched_images, last_patched_images = sample(
    [p["prompt"] for p in prompts_C],
    noises,
    BATCH_SIZE,
    NUM_INFERENCE_STEPS,
    generator,
    DEVICE,
    run_with_cache=False,
    # attn_idx_to_patch=ATTENTIONS_TO_PATCH,
    # timestep_start_patching=TIMESTEP_START_PATCHING,
)

np.save(
    os.path.join(
        SAVE_DIR,
        f"unrelated_non_toxic.npy",
    ),
    patched_images.astype(np.uint8),
)

logging.info("Finito!")