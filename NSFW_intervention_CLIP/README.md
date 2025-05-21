# 🚀 How to Run NSFW-Intervention

Each **CLIP Text Encoder** from the **Stable Diffusion 3 (SD3) pipeline** must be run separately. Below is an example script to train CLIP-L model.

## 🏗 Example Script

Run the following command to train the model:

```bash
python safeclip_training_text.py \
    --clip-backbone 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' \
    --checkpoint-saving-root "/clip_ft_checkpoints_big/" \
    --lr 1e-5 \
    --epoches 12 \
    --train-type "forget"
