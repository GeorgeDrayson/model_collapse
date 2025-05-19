from collections import Counter
from tqdm import tqdm
import numpy as np
from scipy import stats
import operator
import os
import random
from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import textstat
import numpy as np
import mauve

def calculate_mauve(ai_dataset, human_dataset=None, n_samples=1000):
    """
    Calculate MAUVE score between AI and human text.
    If human_dataset is not provided, returns None for mauve score.
    """
    if human_dataset is None:
        return {"mauve": None}

    p_text = ai_dataset['cls_text'] # ai text
    q_text = human_dataset['cls_text'] #human text

    if n_samples < len(q_text):
        random_indices = random.sample(range(len(q_text)), n_samples)

        p_text = [p_text[i] for i in random_indices]
        q_text = [q_text[i] for i in random_indices]

    out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=1, max_text_length=256, verbose=False)

    return {"mauve": out.mauve}

def calculate_flesch_readability(dataset):
    values = [textstat.flesch_reading_ease(text) for text in dataset['cls_text']]
    return {"readability": np.mean(values)} 

def eval_text(text, ngram):
    """Computes unique and total n-grams for a given text and ngram size."""
    token_list = text.strip().split()
    total_num = max(len(token_list) - ngram + 1, 0)
    ngram_set = {" ".join(token_list[i:i + ngram]) for i in range(total_num)}
    return len(ngram_set), total_num

def calculate_diversity_for_sample(example, text_key="cls_text"):
    """Computes diversity score for a single sample in a Hugging Face dataset."""
    text = example[text_key].strip()
    ngram_list = [2, 3, 4]
    
    pred_res_dict = {n: {"unique": 0, "total": 0} for n in ngram_list}

    # Compute n-gram statistics
    for n in ngram_list:
        unique_count, total_count = eval_text(text, n)
        pred_res_dict[n]["unique"] = unique_count
        pred_res_dict[n]["total"] = total_count

    # Calculate repetition rates (rep-n)
    rep_values = {
        f"rep-{n}": round(100 * (1 - pred_res_dict[n]["unique"] / pred_res_dict[n]["total"]), 2) if pred_res_dict[n]["total"] > 0 else 0
        for n in ngram_list
    }

    # Compute final diversity score
    diversity_score = (
        (1 - rep_values["rep-2"] / 100) *
        (1 - rep_values["rep-3"] / 100) *
        (1 - rep_values["rep-4"] / 100)
    )

    # Return updated sample with diversity metrics
    return {**example, "diversity": diversity_score}

def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)

def calculate_self_bleu_score(dataset, n_sample=100):
    random.seed(0)

    all_sentences = dataset["input_ids"]
    smoothing_function = SmoothingFunction().method1

    pool = Pool(processes=os.cpu_count())
    bleu_scores = {}
    for n_gram in range(4, 5):

        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        bleu_scores[n_gram]= list(tqdm(
                        pool.imap_unordered(
                            partial(bleu_i, weights, all_sentences, smoothing_function),
                            random.sample(range(len(all_sentences)), n_sample)),
                        total=n_sample,
                        smoothing=0.0,
                        desc=f"bleu-{n_gram}"))

    results = {}
    for n_gram in range(4,5):
        results[f"self_bleu_{n_gram}"] = sum(bleu_scores[n_gram]) / n_sample

    return results

def calculate_average_length(dataset):

    # Extracting 'input_ids' lengths
    input_lengths = [len(seq) for seq in dataset['input_ids']]

    # Calculating average length
    average_length = np.mean(input_lengths)

    return {"average_length": average_length}