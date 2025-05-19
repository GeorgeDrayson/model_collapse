import numpy as np

def importance_sampling(dataset, seed, max_repeats, bias_factor, upsample_factor, **kwargs):

    # Set seed for reproducibility
    np.random.seed(seed)

    # Compute inverse probability weights
    weights = (1 - np.array(dataset['cls_confidence'])) ** bias_factor

    # Sample indices based on computed weights
    expanded_indices = np.repeat(np.arange(len(dataset)), max_repeats)

    # Sample without replacement
    sampled_indices = np.random.choice(expanded_indices, size=round(len(dataset)*upsample_factor), replace=False, p=weights[expanded_indices] / weights[expanded_indices].sum())

    return dataset.select(sampled_indices)