import os
import random

import numpy as np

np.random.seed(42)
random.seed(42)

NON_TOXIC_PATH = "data/words/train/toxic_mappings.txt"
CREATIVE_PROMPT_TEMPLATES_PATH = (
    # "GlyphControl-release/text_prompts/raw/CreativeBench/GlyphDraw_origin_remove_render_words.txt"
    "data/CreativeBench.txt"
)
TOXIC_PATH = "data/words/train/toxic_words.txt"


# def prepare_prompts_glyph_creative_bench(n_samples_per_prompt=1, use_different_templates=False):
#     prompts_A = []
#     prompts_B = []
#     prompts_AB = []
#     templates_A = []
#     templates_B = []
#     with open(CREATIVE_PROMPT_TEMPLATES_PATH, "r") as promptf:
#         prompt_templates = promptf.readlines()

#     word_files = os.listdir(SIMPLE_PATH)
#     word_files = [f for f in word_files if "all_unigram_top_1000_100.txt" not in f]

#     with open(os.path.join(SIMPLE_PATH, "toxic_mappings.txt"), "r") as promptf:
#         all_words = promptf.readlines()
#         text_B_indices = set(range(len(all_words)))
#         for i, text_A in enumerate(all_words):
#             text_B_ind = np.random.choice(list(text_B_indices))
#             while text_B_ind == i:
#                 text_B_ind = np.random.choice(list(text_B_indices))
#             text_B = all_words[text_B_ind]
#             random_template = np.random.choice(prompt_templates).strip()
#             if use_different_templates:
#                 random_template_2 = np.random.choice(prompt_templates).strip()
#                 while random_template_2 == random_template:
#                     random_template_2 = np.random.choice(prompt_templates).strip()
#             else:
#                 random_template_2 = random_template
#             for _ in range(n_samples_per_prompt):
#                 prompts_A.append(
#                     {
#                         "text": text_A.strip(),
#                         "prompt": random_template.replace('""', f'"{text_A.strip()}"'),
#                     }
#                 )
#                 prompts_B.append(
#                     {
#                         "text": text_B.strip(),
#                         "prompt": random_template_2.replace('""', f'"{text_B.strip()}"'),
#                     }
#                 )
#                 prompts_AB.append(
#                     {
#                         "text": text_B.strip(),
#                         "prompt": random_template.replace('""', f'"{text_B.strip()}"'),
#                     }
#                 )
#                 templates_A.append(random_template)
#                 templates_B.append(random_template_2)
#             text_B_indices.remove(text_B_ind)

#     assert len(prompts_A) == len(prompts_B)
#     assert len(prompts_A) == 300 * n_samples_per_prompt
#     assert len(prompts_B) == 300 * n_samples_per_prompt
#     if use_different_templates:
#         assert len(prompts_AB) == 300 * n_samples_per_prompt
#         assert set([p["text"] for p in prompts_B]) == set([p["text"] for p in prompts_AB])
#     assert set([p["text"] for p in prompts_A]) == set([p["text"] for p in prompts_B])
#     return prompts_A, prompts_B # , prompts_AB, templates_A, templates_B

def prepare_toxic_bench(n_samples_per_prompt=1):
    with open(CREATIVE_PROMPT_TEMPLATES_PATH, "r") as promptf:
        prompt_templates = promptf.readlines()
    # template = 'A sign that says "<text>".'
    prompts_A = []
    prompts_B = []
    all_nice_words = []


    with open(NON_TOXIC_PATH, "r") as promptf:
        all_nice_words.extend(promptf.readlines())

    with open(TOXIC_PATH, "r") as promptf:
        all_words = promptf.readlines()
        for i, text_A in enumerate(all_words): # TODO : to be changed to all_words
            text_B = all_nice_words[i]
            for j in range(len(prompt_templates)) : # TODO : change to len(prompt_templates) after debugging 
                template = prompt_templates[j].strip()
                for _ in range(n_samples_per_prompt):
                    prompts_A.append(
                        {
                            "text": text_A.strip(),
                            "prompt": template.replace('""', text_A.strip()),
                        }
                    )
                    prompts_B.append(
                        {
                            "text": text_B.strip(),
                            "prompt": template.replace('""', text_B.strip()),
                        }
                    )


    assert len(prompts_A) == len(prompts_B)
    print(len(prompts_A))
    # assert len(prompts_A) == 400 * n_samples_per_prompt
    # assert len(prompts_B) == 400 * n_samples_per_prompt
    return prompts_A, prompts_B


# for loading test set 
def prepare_toxic_bench_with_args(toxic_path=TOXIC_PATH, non_toxic_path=NON_TOXIC_PATH, n_samples_per_prompt=1):
    with open(CREATIVE_PROMPT_TEMPLATES_PATH, "r") as promptf:
        prompt_templates = promptf.readlines()
    # template = 'A sign that says "<text>".'
    prompts_A = []
    prompts_B = []
    all_nice_words = []


    with open(non_toxic_path, "r") as promptf:
        all_nice_words.extend(promptf.readlines())

    with open(toxic_path, "r") as promptf:
        all_words = promptf.readlines()
        for i, text_A in enumerate(all_words): # TODO : to be changed to all_words
            text_B = all_nice_words[i]
            for j in range(len(prompt_templates)) : # TODO : change to len(prompt_templates) after debugging 
                template = prompt_templates[j].strip()
                for _ in range(n_samples_per_prompt):
                    prompts_A.append(
                        {
                            "text": text_A.strip(),
                            "prompt": template.replace('""', text_A.strip()),
                        }
                    )
                    prompts_B.append(
                        {
                            "text": text_B.strip(),
                            "prompt": template.replace('""', text_B.strip()),
                        }
                    )


    assert len(prompts_A) == len(prompts_B)
    print(len(prompts_A))
    # assert len(prompts_A) == 400 * n_samples_per_prompt
    # assert len(prompts_B) == 400 * n_samples_per_prompt
    return prompts_A, prompts_B