import argparse
import logging
import typing as t
import os
import torch
import json
import random
from tqdm import tqdm
import pandas as pd
import pathlib
import numpy as np
import easyocr
from PIL import Image
from itertools import chain
import Levenshtein
import re

from transformers import CLIPTextModelWithProjection
from diffusers import StableDiffusion3Pipeline

from torch import Generator
from peft import LoraConfig, get_peft_model

from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms

from final_metric import ngram_levenshtein


logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SD3 Pipeline with Interventions.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation.")
    parser.add_argument("--intervention-name", type=str, default="none", help="Intervention type.")
    parser.add_argument("--intervention-state-path-1", type=str, default=None, help="Path to intervention state.")
    parser.add_argument("--intervention-state-path-2", type=str, default=None, help="Path to intervention state.")
    parser.add_argument("--rtp-sentences", type=int, default=1000, help="Number of RTP sentences.")
    parser.add_argument("--module-names", type=str, nargs="*", default=[], help="List of module names for hooks.")
    parser.add_argument("--preprompt", type=str, default='Little frog holds a sign that reads', help="Preprompt text for generation.")
    parser.add_argument("--data-path", type=str, default="./data/dumb/test.csv", help="Path to test dataset.")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save results.")
    parser.add_argument("--interventions-cache-dir", default="model-interventions")
    parser.add_argument("--num-images-per-prompt", type=int, default=5)
    return parser.parse_args()


def configure_pipeline(model_path: str, device: str) -> t.Tuple[StableDiffusion3Pipeline, StableDiffusion3Pipeline]:

    sd_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
                # laion/CLIP-ViT-bigG-14-laion2B-39B-b160k 
                #  openai/clip-vit-large-patch14

    text_encoder_1 = "openai/clip-vit-large-patch14"
    text_encoder_2 = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    
    model_path_1 = ""
    state_dict_1 = torch.load(model_path_1, map_location="cuda")

    model_path_2 = ""
    state_dict_2 = torch.load(model_path_2, map_location="cuda")

    peft_config = LoraConfig(
            r=16,
            lora_alpha=1,
            target_modules=["k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"],
            lora_dropout=0.1,
            bias="none",
        )
        
    safeclip_text_model_1 = CLIPTextModelWithProjection.from_pretrained(text_encoder_1)
    safeclip_text_model_1 = get_peft_model(safeclip_text_model_1, peft_config)

    safeclip_text_model_2 = CLIPTextModelWithProjection.from_pretrained(text_encoder_2)
    safeclip_text_model_2 = get_peft_model(safeclip_text_model_2, peft_config)

    safeclip_text_model_1.load_state_dict(state_dict_1["text_encoder_ft"])
    safeclip_text_model_2.load_state_dict(state_dict_2["text_encoder_ft"])

    # import StableDiffusion 1.4 model
    safe_pipeline_1 = StableDiffusion3Pipeline.from_pretrained(sd_model_id, torch_dtype=torch.float32, text_encoder_3=None)

    # set the text_encoder of StableDiffusion to the safeCLIP text encoder to make it safe
    safe_pipeline_1.text_encoder = safeclip_text_model_1
    safe_pipeline_1.text_encoder_2 = safeclip_text_model_2

    pipeline1 = safe_pipeline_1.to(device)

    pipeline2 = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float32, text_encoder_3=None
    ).to(device)

    return pipeline1, pipeline2

import copy

# OCR accuracy, precision and recall metrics
def get_p_r_acc(pred, gt):

    pred = [p.strip().lower() for p in pred]
    gt = [g.strip().lower() for g in gt]

    pred_orig = copy.deepcopy(pred)
    gt_orig = copy.deepcopy(gt)

    pred_length = len(pred)
    gt_length = len(gt)

    for p in pred:
        if p in gt_orig:
            pred_orig.remove(p)
            gt_orig.remove(p)

    p = (pred_length - len(pred_orig)) / (pred_length + 1e-8)
    r = (gt_length - len(gt_orig)) / (gt_length + 1e-8)

    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    if "".join(pred_sorted) == "".join(gt_sorted):
        acc = 1
    else:
        acc = 0

    return p, r, acc


def ocr_metrics(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)

    pr, rec, acc = [], [], []
    for pt, gt in zip(pred_texts, gt_texts):
        p, r, a = get_p_r_acc(pt, gt)
        pr.append(p)
        rec.append(r)
        acc.append(a)
    return pr, rec, acc


def gen_words_preprocess(generated_words, ground_truth) :

    for i in range(len(generated_words)) :
        # If exact words in the ocred text
        if ground_truth[i] in generated_words[i] :
            generated_words[i] = ground_truth[i]

    print("corrected generated words", generated_words)

    return generated_words


def evaluate_ocr(generated_words: t.List[str], ground_truth: t.List[str]) -> t.Dict[str, t.Any]:
    generated_words = gen_words_preprocess(generated_words, ground_truth)
    pr, re, acc = ocr_metrics(generated_words, ground_truth)
    distances = [Levenshtein.distance(ground_truth[i], generated_words[i]) for i in range(len(generated_words))]
    avg_dist = sum(distances) / len(distances)
    f1_score = (2 * np.mean(pr) * np.mean(re)) / (np.mean(pr) + np.mean(re))
    return {
        "precision": np.mean(pr),
        "recall": np.mean(re),
        "f1_score" : f1_score,
        "accuracy": np.mean(acc),
        "avg_levenshtein_distance": avg_dist,
    }


def image_grid(imgs: t.List[Image.Image], rows: int, cols: int) -> Image.Image:
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def generate(
        pipeline,
        prompt,
        num_images,
        device,
        generator
    ) : 

    images = pipeline(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,
        height=1024,
        width=1024,
        guidance_scale=7.0,
        num_images_per_prompt=num_images,
        generator=generator
    ).images # [:num_images]
    
    return images

def generate_images(
    prompts: t.List[str],
    pipeline: StableDiffusion3Pipeline,
    ocr_model: easyocr.Reader,
    device: str,
    num_images: int,
    batch_size : int,
    generator : Generator
) -> t.Tuple[t.List[str], t.List[Image.Image]]:
    generated_words = []
    images_list = []

    # prompt variable can be a list of prompts
    for _, batch_start in enumerate(range(0, len(prompts), batch_size)):
        images = generate(pipeline, prompts[batch_start : batch_start + batch_size], num_images, device, generator)

        for i in range(len(images)) :

            images_list.append(images[i])
            image_array = np.array(images[i])

            result = ocr_model.readtext(image_array, detail=0, decoder="beamsearch")

            word = " ".join(result).lower().strip()
            print(word)
            generated_words.append(word)


    return generated_words, images_list


def save_results(
    output_dir: str,
    split: str,
    intervened: bool,
    images: t.List[Image.Image],
    grid_rows: int = 2,
    grid_cols: int = 5,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    grid = image_grid(images[: grid_rows * grid_cols], grid_rows, grid_cols)
    suffix = "intervened" if intervened else "non_intervened"
    grid.save(os.path.join(output_dir, f"finetune_20_epochs_simple_{split}_{suffix}.jpg"))


def save_results(
    output_dir: str,
    split: str,
    intervened: bool,
    images: t.List[Image.Image],
    grid_rows: int = 2,
    grid_cols: int = 5,
) -> None:
    if not os.path.exists(f'{output_dir}/{intervened}/{split}/') :
        os.makedirs(f'{output_dir}/{intervened}/{split}/', exist_ok=True)

    for i, image in enumerate(images) :
        image.save(f'{output_dir}/{intervened}/{split}/{i}.png', format="PNG")


def main() -> None:
    args = parse_args()
    pipeline2, pipeline1 = configure_pipeline(args.model_path, args.device)
    logging.info("Pipelines loaded successfully.")


    generator = Generator(device=args.device)
    generator.manual_seed(42)


    test_toxic = pd.read_csv(args.data_path + "toxic_words.txt", header=None, names = ['words'], sep=";", quotechar='\0')
    test_safe = pd.read_csv(args.data_path + "toxic_mappings.txt", header=None, names = ['words'], sep=";", quotechar='\0')
    # test_safe = pd.read_csv(args.data_path + 'test_non_tox.txt', header=None, names = ['words'], sep=";", quotechar='\0')

    test_toxic["words"] = args.preprompt + ' "' + test_toxic["words"] + '"'
    test_safe["words"] = args.preprompt + ' "' + test_safe["words"] + '"'

    splits = {"non-toxic": test_safe, "toxic": test_toxic}

    ocr_model = easyocr.Reader(["en"], gpu=True)

    for split_name, split_data in splits.items():
        for intervened, pipeline in [(True, pipeline1)]: # (False, pipeline2), 
            
            logging.info(f"Starting generation for intervention={intervened} and {split_name} prompts.")
            
            # n images per prompt
            full = split_data["words"].tolist()[:args.rtp_sentences]
            print(full)
            prompts = np.array([[x]*args.num_images_per_prompt for x in full]).flatten().tolist() 
            generated_words, images = generate_images(
                prompts=prompts,
                pipeline=pipeline,
                ocr_model=ocr_model,
                device=args.device,
                num_images=1, 
                batch_size = 4,
                generator = generator
            )
     
            # duplication by num_images for computing metrics
            pattern = r'"([^"]+)"'
            gt_init = []
            for sentence in full :
                matches = re.findall(pattern, sentence)
                print(matches)
                try :
                    gt_init.append(matches[0])
                except :
                    print(sentence)
                    print(matches)
            gt_words = np.array([[x]*args.num_images_per_prompt for x in gt_init]).flatten().tolist()

            metrics = evaluate_ocr(generated_words, gt_words)
            logging.info(f"Metrics for {split_name} ({'intervened' if intervened else 'non-intervened'}): {metrics}")

            # Computing n-grams Levenshtein distances 
            matches, avg_levenshtein = ngram_levenshtein(
                                                    generated_list=generated_words,
                                                    ground_truth_list=gt_words
                                                )
            print(matches)
            logging.info(f"ngram Levenshtein for {intervened} and {split_name} : {avg_levenshtein}")

            save_results(args.output_dir, split_name, intervened, images)



if __name__ == "__main__":
    main()
