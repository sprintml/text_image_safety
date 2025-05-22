# NSFW-Intervention

**NSFW-Intervention** is a fine-tuning pipeline designed to reduce toxic content generation in diffusion-based models. It supports fine-tuning of **Stable Diffusion v3**, **DeepFloyd IF**, and **SDXL** using DreamBooth-based LoRA methods.

---

## 🛠️ Project Setup

Clone and install the required packages from HuggingFace's `diffusers` library:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .

accelerate config default
```

---

## 📁 Dataset Requirements

Fine-tuning requires **two image-folder-type datasets used together**, both with prompts:

1. **Toxic Dataset**  
   A folder containing the toxic images from ToxicBench and their associated prompts.

2. **Mapping Dataset**  
   A corresponding folder containing the benign image mappings and their related prompts.

Make sure both datasets follow the `image_folder` structure with appropriate metadata (e.g., `metadata.jsonl` or CSV prompt mapping).

---

## 🧪 Fine-Tuning

The main training scripts available are:

- `sd3.py` for **Stable Diffusion v3**
- `deepfloyd.py` for **DeepFloyd IF**
- `sdxl.py` for **Stable Diffusion XL**

Each script shares a similar CLI pattern and requires minimal modifications.

### Update Paths in Training Scripts

**Training Dataset Path:** In each script, update the training data path to point to your toxic and benign dataset folders. The corressponding parsed arguments are: 
- `args.toxic_dataset`
- `args.target_dataset`

### Example Execution (`sd3.py`)

```bash
python sd3.py  --pretrained_model_name_or_path=$MODEL_NAME   --toxic_dataset=/path/to/toxic_dataset   --target_dataset=/path/to/target_dataset   --enable_xformers_memory_efficient_attention   --resolution=512   --train_batch_size=$BATCH_SIZE   --val_batch_size=2   --gradient_accumulation_steps=1   --gradient_checkpointing   --num_train_epochs=100   --learning_rate=$LR   --lr_scheduler="constant"   --lr_warmup_steps=0   --mixed_precision="fp16"   --report_to="wandb"   --checkpointing_steps=500   --output_dir=/path/to/output_model   --seed=42   --target_layer=10   --validation_epochs=10   --center_crop  
```

Make sure to adapt this command for `deepfloyd.py` and `sdxl.py` accordingly.

---

## 📊 Visualization

WandB is integrated for experiment tracking. Log in before running any training:

```bash
wandb login
```

---

## ⚙️ Notes

- Both datasets must include prompts.
- Dataset format must be compatible with HuggingFace `datasets.load_dataset('imagefolder')` or custom loaders in the repo.
- Distributed training is supported for all models with `accelerate` library.

---
