import pandas as pd
import os
import numpy as np 
import random 
import argparse

def prepare_prompts_glyph_creative_bench(word_path):
    prompts = {}

    with open('CreativeBench.txt', "r") as promptf:
        prompt_templates = promptf.readlines()

    word_files = os.listdir(word_path)
    word_files = [f for f in word_files]

    for word_file in word_files:
        prompts[word_file] = []
        with open(os.path.join(word_path, word_file), "r") as promptf:
            all_words = promptf.readlines()
        for word in all_words :
            for prompt in prompt_templates :
                prompts[word_file].append(prompt.replace('""', f'"{word.strip()}"'))

        # Here want every dataset to have the same template mapping, even harmless words
        random.seed(1)
        
        random.shuffle(prompts[word_file])
        

    return prompts

if __name__=="__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, description='Seeding for shuffling')
    parser.add_argument('--word-path')

    word_path = "./final_train_values/"
    prompts= prepare_prompts_glyph_creative_bench(word_path)

    if not os.path.exists('./final_train') :
        os.mkdir('./final_train')
    for word_file in prompts.keys() :
        with open(f"./final_train/{word_file}", 'w') as file:
            for item in prompts[word_file]:
                file.write(item)