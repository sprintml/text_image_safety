import os
from pathlib import Path
import time
import torch
import json
import ast
import math
import wandb
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model

from training.dataset.toxic_dataset import ToxicDataset
from training.utils.argumentparser import parse_arguments
from training.new_train_text_fp16 import train_text_encoder
from training.utils.logger import WandbLogger
from training.losses import CLIPLoss_Positive, CosineDistance

from eval_in_training import evaluation_in_training


def main(args):
    hyperparameters = {
        'clip_backbone': args.clip_backbone,
        'lora_r': args.lora_r,
        'epoches': args.epoches,
        'lr': args.lr,
        'wandb_activated': args.wandb_activated,
        'wandb_config': ast.literal_eval(args.wandb_config),
        'lambdas' : torch.cat([torch.tensor(x, dtype=torch.float32)[(None,)+(...,)] for x in ast.literal_eval(args.lambdas)]),
        'batch_size': args.bs,
        'device': args.device,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'initial_patience': args.initial_patience,
        'checkpoint_saving_root': args.checkpoint_saving_root,
        'resume': args.resume,
        'resume_checkpoints_path': args.resume_checkpoints_path,
        'wandb_run_id': args.wandb_run_id,
        'debug': args.debug,
        'train_type': args.train_type,
        'slurm_id' : args.slurm_id
    }
    
    debug = hyperparameters['debug']
    clip_backbone = hyperparameters['clip_backbone']
    lora_r = hyperparameters['lora_r']
    epoches = hyperparameters['epoches']
    lr = hyperparameters['lr']
    wandb_activated = hyperparameters['wandb_activated']
    wandb_config = hyperparameters['wandb_config']
    wandb_run_id = hyperparameters['wandb_run_id']
    lambdas = hyperparameters['lambdas']
    batch_size = hyperparameters['batch_size']
    device = hyperparameters['device']
    gradient_accumulation_steps = hyperparameters['gradient_accumulation_steps']
    initial_patience = hyperparameters['initial_patience']
    checkpoint_saving_path = hyperparameters['checkpoint_saving_root']
    resuming_wandb_run = False if wandb_run_id == 'None' else True
    resume = hyperparameters['resume']
    resume_checkpoints_path = hyperparameters['resume_checkpoints_path']
    train_type = hyperparameters['train_type']
    slurm_id = hyperparameters['slurm_id']

    if 'leonardo' in str(Path(__file__)) or debug:
        os.environ["WANDB_MODE"] = "offline"

    if wandb_activated:
        if not resuming_wandb_run:
            run = wandb.init(
                settings=wandb.Settings(start_method="fork"),
                reinit=True, config=hyperparameters, **wandb_config
            )
        else:
            run = wandb.init(
                settings=wandb.Settings(start_method="fork"),
                reinit=True, config=hyperparameters, **wandb_config,
                resume='must', id=wandb_run_id
            )
        print(f'Wandb ID: {run.id}')
        wandb_logger = WandbLogger(run)
    else:
        wandb_logger = None
        run = None

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=1,
        target_modules=["k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"],
        lora_dropout=0.1,
        bias="none",
    )


    tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)

    text_encoder_original = CLIPTextModelWithProjection.from_pretrained(clip_backbone) # , torch_dtype=torch.float16)
    text_encoder_ft = get_peft_model(text_encoder_original, peft_config)

    training_dataset = ToxicDataset(data_folder="./final_train/")
    validation_dataset = ToxicDataset(data_folder="./final_test/")


    # Normalizing
    lambdas = lambdas / lambdas.sum()

    if not resume:
        # create a unique dir where to save checkpoints
        job_id = os.environ.get("SLURM_JOB_ID")
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        proc_id = os.environ.get("SLURM_PROCID")
        pid = os.getpid()
        timestamp = math.floor(time.time())
        unique_dir = f"job_{job_id}_task_{task_id}_proc_{proc_id}_pid_{pid}_time_{timestamp}" 
        checkpoint_saving_path = Path(checkpoint_saving_path) / unique_dir

        if not Path(checkpoint_saving_path).exists():
            try:
                os.makedirs(checkpoint_saving_path)
            except Exception as e:
                print(e)

        _hyp = {k:str(v) for k,v in hyperparameters.items()}
        with open(Path(checkpoint_saving_path / 'config'), 'w') as f:
            json.dump(_hyp, f)

    else:
        checkpoint_saving_path = resume_checkpoints_path
        assert Path(checkpoint_saving_path).exists(), ValueError(f'{checkpoint_saving_path} is not an existing dir.')

    checkpoint_saving_path = Path(checkpoint_saving_path) if type(checkpoint_saving_path) == str else checkpoint_saving_path
    if wandb_activated and run is not None:
        run.log({'checkpoint_saving_path': str(checkpoint_saving_path)})

    print('Training...')
    train_text_encoder(
        text_encoder_ft=text_encoder_ft,
        text_encoder_original=text_encoder_original,
        tokenizer=tokenizer,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        loss_function=CosineDistance(),
        lambdas=lambdas,
        batch_size=batch_size,
        lr=lr,
        epochs=epoches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        patience_threshold=25,
        wandb_activated=wandb_activated,
        run=run,
        device=device,
        checkpoint_path=checkpoint_saving_path,
        resume=resume,
        debug=debug,
        wandb_logger=wandb_logger,
        train_type=train_type,
    )


    if not os.path.exists("./model_ids/") :
        os.mkdir("./model_ids/")

    if clip_backbone == "openai/clip-vit-large-patch14" :
        model_size = "small"
    else :
        model_size = "big"
        
    with open(f"./model_ids/{slurm_id}_{model_size}.csv", 'w') as file :
        file.write(f"{unique_dir}")


if __name__ == '__main__':
    main(parse_arguments())