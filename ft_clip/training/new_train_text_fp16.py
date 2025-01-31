import os
import itertools
import torch
from torch import cuda
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from peft import PeftModel

from training.dataset.toxic_dataset import ToxicDataset
from training.losses import CosineDistance
from training.utils.logger import WandbLogger, summarize
from training.new_validation_text_fp16 import validate
from training.utils.checkpointing import CheckpointManager

from torch.cuda.amp import autocast, GradScaler


@torch.enable_grad()
def train_text_encoder(
    text_encoder_ft: PeftModel,
    text_encoder_original: CLIPTextModelWithProjection,
    tokenizer: CLIPTokenizer,
    train_dataset: ToxicDataset,
    validation_dataset: ToxicDataset,
    loss_function: CosineDistance,
    lambdas=1.0,
    batch_size=32,
    lr=1e-5,
    epochs=10,
    gradient_accumulation_steps=1,
    patience_threshold=5,
    wandb_activated=False,
    run=None,
    device='cuda',
    checkpoint_path='',
    resume=False,
    debug=False,
    wandb_logger: WandbLogger | None = None,
    train_type='all'
):
    optimizer = torch.optim.Adam(
        (p for n, p in text_encoder_ft.named_parameters() if 'lora' in n), lr=lr
    )

    for n, p in text_encoder_ft.named_parameters():
        p.requires_grad = 'lora' in n
    
    checkpoint_manager = CheckpointManager(
        checkpoint_path, text_encoder_ft=text_encoder_ft, optimizer=optimizer
    )

    start_epoch, best_loss, patience = (-1, float('inf'), patience_threshold)
    
    if resume:
        start_epoch, best_loss, _, patience = checkpoint_manager.resume(mode='last')
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    text_encoder_ft.to(device)
    text_encoder_original.to(device)

    training_start = cuda.Event(enable_timing=True)
    validation_start = cuda.Event(enable_timing=True)
    training_end = cuda.Event(enable_timing=True)
    validation_end = cuda.Event(enable_timing=True)

    training_start.record()

    scaler = GradScaler()

    epsilon = torch.tensor(1e-10, device=device, dtype=torch.float32)

    for epoch in range(start_epoch + 1, epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

        text_encoder_ft.train()
        text_encoder_original.eval()

        for idx, (nsfw_caption, safe_caption, harmless_caption) in enumerate(tqdm(train_loader)):


            lambdas = lambdas.to(device).float()

            text_safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)
            text_nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)
            text_harmless_ids = tokenizer(harmless_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)

            # model_text_safe = text_encoder_ft(**text_safe_ids).text_embeds
            model_text_nsfw = text_encoder_ft(**text_nsfw_ids).text_embeds
            model_text_harmless = text_encoder_ft(**text_harmless_ids).text_embeds

            with torch.no_grad():
                # reference_text_safe = text_encoder_original(**text_safe_ids).text_embeds
                reference_text_nsfw = text_encoder_original(**text_nsfw_ids).text_embeds
                reference_text_harmless = text_encoder_original(**text_harmless_ids).text_embeds


            loss_harmless = torch.clamp(loss_function(model_text_harmless, reference_text_harmless).mean(dim=0), min=epsilon)
            loss_safe = torch.log(loss_harmless)

            loss_nsfw = torch.clamp(loss_function(model_text_nsfw, reference_text_nsfw).mean(dim=0), min=epsilon)
            loss_nsfw = -torch.log(loss_nsfw)

        
            if train_type == 'forget' :
                training_loss = loss_nsfw
            else : 
                losses = torch.cat(
                                [
                                    x[(None,)+(...,)] \
                                        for x in (loss_safe, loss_nsfw)
                                ]
                            )
                training_loss = (lambdas @ losses[(None,)+(...,)].T) / lambdas.numel()

            training_loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(text_encoder_ft.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad()

                
            if wandb_activated and run is not None:
                wandb_logger.log_training_iteration(idx, loss_safe, loss_nsfw, training_loss)

            if debug:
                break

        training_end.record()
        cuda.synchronize()
        training_time = training_start.elapsed_time(training_end)/1000
        validation_start.record()

        val_loss, val_loss_safe, val_loss_nsfw, val_loss_harmless = validate(
            text_encoder_ft, text_encoder_original, tokenizer, validation_dataset, loss_function,
            lambdas, batch_size, False, None, device, debug, wandb_logger
        )

        validation_end.record()
        cuda.synchronize()
        validation_time = validation_start.elapsed_time(validation_end)/1000

        if val_loss < best_loss:
            best_loss = val_loss
            patience = patience_threshold
            checkpoint_manager.best('validation-loss', epoch, best_loss, 0, patience)
        else:
            patience -= 1

        if patience == 0:
            break


        if epoch - 1 % 5 == 0:        
            checkpoint_manager.step(epoch, best_loss, 0, patience)
        summarize(epoch, patience, training_loss, val_loss, val_loss_safe, val_loss_nsfw, val_loss_harmless, 0, 0, best_loss, training_time, validation_time, checkpoint_path)
