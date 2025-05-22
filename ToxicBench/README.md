# ToxicBench

![ToxicBench Evaluation Pipeline](ToxicBench.png)

## Installation Guide

Clone the repo and install the required libraries for this repo :

```bash
git clone https://github.com/sprintml/ToxicBench
cd ToxicBench
pip install -r requirements.txt
```


## How To Use The Benchmark

For generating the dataset :

```bash
cd ./data
python preprocess.py \
    --seed 1 \
    --word-path "./words/train" \
    --output-path "./train" 
python preprocess.py \
    --seed 1 \
    --word-path "./words/test" \
    --output-path "./test" 
```

For evaluating, an example file ```example_sd3.py``` is provided. Test evaluation can be done with :

```bash
python example_sd3.py \
    --model-path-1 "stabilityai/stable-diffusion-3-medium-diffusers" \
    --model-path-2 "stabilityai/stable-diffusion-3-medium-diffusers" \
    --device "cuda" \
    --num-samples 20 \
    --data-path "./data/words/test" \
    --output-dir "./output-test" \
    --batch-size 4 \
    --seed 42 \
    --save-images True \
    --ocr-model "easyocr"
```
