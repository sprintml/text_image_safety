# Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images

## Abstract
State-of-the-art Diffusion Models (DMs) produce highly realistic images. While prior work has successfully mitigated Not Safe For Work (NSFW) content in the visual domain, we identify a novel threat: the generation of NSFW text embedded within images. This includes offensive language, such as insults, racial slurs, and sexually explicit terms, posing significant risks to users. We show that all state-of-the-art DMs (e.g., SD3, SDXL, Flux, DeepFloyd IF) are vulnerable to this issue. Through extensive experiments, we demonstrate that existing mitigation techniques, effective for visual content, fail to prevent harmful text generation while substantially degrading benign text generation. As an initial step toward addressing this threat, we introduce a novel fine-tuning strategy that targets only the text-generation layers in DMs. Therefore, we construct a safety fine-tuning dataset by pairing each NSFW prompt with two images: one with the NSFW term, and another where that term is replaced with a carefully crafted benign alternative while leaving the image unchanged otherwise. By training on this dataset, the model learns to avoid generating harmful text while preserving benign content and overall image quality. Finally, to advance research in the area, we release ToxicBench, an open-source benchmark for evaluating NSFW text generation in images. It includes our curated fine-tuning dataset, a set of harmful prompts, new evaluation metrics, and a pipeline that assesses both NSFW-ness and text and image quality. Our benchmark aims to guide future efforts in mitigating NSFW text generation in text-to-image models---contributing to their safe deployment.

## 📂 Project Structure
This repository contains scripts for different parts of the project:
- **`NSFW_intervention/`** – Implementation of **NSFW-Intervention**.
- **`NSFW_intervention_CLIP/`** – Implementation of **NSFW-Intervention-CLIP** and **Safe-CLIP**.
- **`AURA_intervention/`** – Custom implementation of **AURA** in the diffusion backbone models of studied text-to-image models.
- **`ToxicBench/`** - Our new dataset and evaluation pipeline.

## 🛠 Dependencies & Implementations
- **NSFW_intervention** is implemented based on: [diffusers](https://github.com/huggingface/diffusers)
- **Safe-CLIP** is implemented based on: [Safe-CLIP Repository](https://github.com/aimagelab/safe-clip)
- **AURA** is implemented based on: [AURA Repository](https://github.com/apple/ml-aura)
