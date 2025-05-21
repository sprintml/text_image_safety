import re
from transformers import CLIPTokenizer
from Levenshtein import distance as levenshtein_distance

# Load CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def preprocess_text(text):
    """Clean and normalize text by removing special characters and extra spaces."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()

def get_token_count(text):
    """Returns the number of tokens in the text using CLIP tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def generate_ngrams(text, min_n, max_n):
    """Generate n-grams from min_n to max_n based on tokenized word count."""
    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()

    ngram_candidates = []
    for n in range(min_n, max_n + 1):
        if n > len(words):
            break  # Avoid generating n-grams larger than available words
        ngram_candidates.extend([' '.join(words[i:i + n]) for i in range(len(words) - n + 1)])
    
    return ngram_candidates

def levenshtein_match(generated_text, ground_truth):
    """Find the substring with the closest Levenshtein distance for varying n-gram sizes."""
    ground_truth_tokens = get_token_count(ground_truth)
    ngram_candidates = generate_ngrams(generated_text, min_n=1, max_n=ground_truth_tokens + 1)

    # Rank candidates by Levenshtein distance
    ranked_matches = sorted(ngram_candidates, key=lambda x: levenshtein_distance(x, ground_truth))
    print(ranked_matches)
    best_match = ranked_matches[0]
    best_score = levenshtein_distance(best_match, ground_truth)
    return best_match, best_score

def ngram_levenshtein(generated_list, ground_truth_list):
    """Process multiple generated strings and compute average Levenshtein distance."""
    assert len(generated_list) == len(ground_truth_list), "Lists must have the same length."

    total_distance = 0
    results = []

    for generated, truth in zip(generated_list, ground_truth_list):
        # handling generated empty strings 
        if generated == '':
            best_match, distance = '', len(truth)
        else : 
            best_match, distance = levenshtein_match(generated, truth)
        total_distance += distance
        results.append((truth, best_match, distance))

    average_distance = total_distance / len(generated_list)
    return results, average_distance