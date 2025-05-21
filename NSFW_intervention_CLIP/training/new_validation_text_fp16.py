import os
import torch
from tqdm import tqdm
from training.losses import CLIPLoss_Positive, CosineDistance
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader

from training.dataset.toxic_dataset import ToxicDataset
from training.utils.logger import WandbLogger

from torch.cuda.amp import autocast


@torch.inference_mode()
def validate(
    text_encoder_ft: PeftModel,
    text_encoder_original: CLIPTextModelWithProjection,
    tokenizer: CLIPTokenizer,
    validation_dataset: ToxicDataset,
    distance_loss_function: CosineDistance,
    lambdas=(1,1,1,1),
    batch_size=32,
    wandb_activated=False,
    run=None,
    device='cuda',
    debug=False,
    wandb_logger: WandbLogger | None = None
):
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=len(os.sched_getaffinity(0)))

    text_encoder_ft.eval().to(device)
    text_encoder_original.eval().to(device)

    # lambdas = lambdas/lambdas.numel()
    lambdas = torch.tensor(lambdas, device=device).float()
    # print(lambdas.device)

    text_safe_loss_cumulative = 0
    text_nsfw_loss_cumulative = 0
    text_harmless_loss_cumulative = 0
    validation_loss = 0

    epsilon = torch.tensor(1e-10, device=device, dtype=torch.float32)

    for (nsfw_caption, safe_caption, harmless_caption) in tqdm(validation_dataloader):

        # with autocast() :

            this_batch_size = len(safe_caption)
            
            # text_safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)
            text_nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)
            text_harmless_ids = tokenizer(harmless_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)


            # model_text_safe_embeddings = text_encoder_ft(**text_safe_ids).text_embeds
            model_text_nsfw_embeddings = text_encoder_ft(**text_nsfw_ids).text_embeds
            model_text_harmless = text_encoder_ft(**text_harmless_ids).text_embeds

            # reference_text_embeddings = text_encoder_original(**text_safe_ids).text_embeds
            reference_text_harmless = text_encoder_original(**text_harmless_ids).text_embeds
            reference_text_nsfw = text_encoder_original(**text_nsfw_ids).text_embeds

            text_loss_harmless = torch.clamp(distance_loss_function(model_text_harmless, reference_text_harmless).mean(dim=0), min=epsilon)
            text_loss_harmless = -torch.log(text_loss_harmless)

            text_nsfw_loss = torch.clamp(distance_loss_function(model_text_nsfw_embeddings, reference_text_nsfw).mean(dim=0), min=epsilon)
            text_nsfw_loss = torch.log(text_nsfw_loss)


            # # text_safe_loss_cumulative += (text_safe_loss * this_batch_size).cpu()
            text_nsfw_loss_cumulative += torch.round((text_nsfw_loss * this_batch_size)).cpu()
            text_harmless_loss_cumulative += torch.round((text_loss_harmless * this_batch_size)).cpu()


            losses = torch.cat(
                        [
                            x[(None,)+(...,)] \
                                for x in (text_nsfw_loss, text_loss_harmless)
                        ]
                    )
            validation_loss += torch.round(this_batch_size * (lambdas @ losses[(None,)+(...,)].T) / lambdas.numel())

            if debug:
                break 

    if wandb_activated and run is not None:
        wandb_logger.log_validation(len(validation_dataset), text_safe_loss_cumulative, text_nsfw_loss_cumulative, validation_loss, this_batch_size)
    
    return validation_loss / len(validation_dataset), text_safe_loss_cumulative / len(validation_dataset), text_nsfw_loss_cumulative / len(validation_dataset), text_harmless_loss_cumulative / len(validation_dataset)
