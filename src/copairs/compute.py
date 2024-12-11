import itertools
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from tqdm.autonotebook import tqdm


def parallel_map(par_func: Callable[[int], None], items: np.ndarray) -> None:
    """Execute a function in parallel over a list of items.

    This function uses a thread pool to process items in parallel, with progress
    tracking via `tqdm`. It is particularly useful for batch operations that benefit
    from multithreading.

    Parameters:
    ----------
    par_func : Callable
        A function to execute for each item. It should accept a single argument
        (an item index or value).
    items : np.ndarray
        An array or list of items to process.
    """
    # Total number of items to process
    num_items = len(items)

    # Determine the number of threads to use, limited by CPU count
    pool_size = min(num_items, os.cpu_count())

    # Calculate chunk size for dividing work among threads
    chunksize = num_items // pool_size

    # Use a thread pool to execute the function in parallel
    with ThreadPool(pool_size) as pool:
        # Map the function to items with unordered execution for better efficiency
        tasks = pool.imap_unordered(par_func, items, chunksize=chunksize)

        # Display progress using tqdm
        for _ in tqdm(tasks, total=len(items), leave=False):
            pass


def batch_processing(
    pairwise_op: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    """Decorator for adding batch processing to pairwise operations.

    This function wraps a pairwise operation to process data in batches, enabling
    efficient computation and multithreading when working with large datasets.

    Parameters:
    ----------
    pairwise_op : Callable
        A function that computes pairwise operations (e.g., similarity or distance)
        between two arrays of features.

    Returns:
    -------
    Callable
        A wrapped function that processes pairwise operations in batches.

    """

    def batched_fn(feats: np.ndarray, pair_ix: np.ndarray, batch_size: int):
        # Total number of pairs to process
        num_pairs = len(pair_ix)

        # Initialize an empty result array to store pairwise operation results
        result = np.empty(num_pairs, dtype=np.float32)

        def par_func(i):
            # Extract the features for the current batch of pairs
            x_sample = feats[pair_ix[i : i + batch_size, 0]]
            y_sample = feats[pair_ix[i : i + batch_size, 1]]

            # Compute pairwise operations for the current batch
            result[i : i + len(x_sample)] = pairwise_op(x_sample, y_sample)

        # Use multithreading to process the batches in parallel
        parallel_map(par_func, np.arange(0, num_pairs, batch_size))

        return result

    return batched_fn


def pairwise_corr(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute the Pearson correlation coefficient for paired rows of two matrices.

    Parameters:
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns:
    -------
    np.ndarray
        A 1D array of Pearson correlation coefficients for each row pair in
        `x_sample` and `y_sample`.
    """
    # Compute the mean for each row
    x_mean = x_sample.mean(axis=1, keepdims=True)
    y_mean = y_sample.mean(axis=1, keepdims=True)

    # Center the rows by subtracting the mean
    x_center = x_sample - x_mean
    y_center = y_sample - y_mean

    # Compute the numerator (dot product of centered vectors)
    numer = (x_center * y_center).sum(axis=1)

    # Compute the denominator (product of vector magnitudes)
    denom = (x_center**2).sum(axis=1) * (y_center**2).sum(axis=1)
    denom = np.sqrt(denom)

    # Calculate correlation coefficients
    corrs = numer / denom
    return corrs


def pairwise_cosine(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for paired rows of two matrices.

    Parameters:
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns:
    -------
    np.ndarray
        A 1D array of cosine similarity scores for each row pair in `x_sample` and `y_sample`.
    """
    # Normalize each row to unit vectors
    x_norm = x_sample / np.linalg.norm(x_sample, axis=1)[:, np.newaxis]
    y_norm = y_sample / np.linalg.norm(y_sample, axis=1)[:, np.newaxis]

    # Compute the dot product of normalized vectors
    c_sim = np.sum(x_norm * y_norm, axis=1)
    return c_sim


def pairwise_abs_cosine(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute the absolute cosine similarity for paired rows of two matrices.

    Parameters:
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns:
    -------
    np.ndarray
        Absolute values of cosine similarity scores.
    """
    return np.abs(pairwise_cosine(x_sample, y_sample))


def pairwise_euclidean(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """
    Compute the inverse Euclidean distance for paired rows of two matrices.

    Parameters:
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns:
    -------
    np.ndarray
        A 1D array of inverse Euclidean distance scores (scaled to range 0-1).
    """
    # Compute Euclidean distance and scale to a range of 0 to 1
    e_dist = np.sqrt(np.sum((x_sample - y_sample) ** 2, axis=1))
    return 1 / (1 + e_dist)


def pairwise_manhattan(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute the inverse Manhattan distance for paired rows of two matrices.

    Parameters:
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns:
    -------
    np.ndarray
        A 1D array of inverse Manhattan distance scores (scaled to range 0-1).
    """
    m_dist = np.sum(np.abs(x_sample - y_sample), axis=1)
    return 1 / (1 + m_dist)


def pairwise_chebyshev(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
    """Compute the inverse Chebyshev distance for paired rows of two matrices.

    Parameters:
    ----------
    x_sample : np.ndarray
        A 2D array where each row represents a profile.
    y_sample : np.ndarray
        A 2D array of the same shape as `x_sample`.

    Returns:
    -------
    np.ndarray
        A 1D array of inverse Chebyshev distance scores (scaled to range 0-1).
    """
    c_dist = np.max(np.abs(x_sample - y_sample), axis=1)
    return 1 / (1 + c_dist)


def get_distance_fn(distance):
    """
    Retrieve a distance metric function based on a string identifier or custom callable.

    This function provides flexibility in specifying the distance metric to be used
    for pairwise similarity or dissimilarity computations. Users can choose from a
    predefined set of metrics or provide a custom callable.

    Parameters:
    ----------
    distance : str or callable
        The name of the distance metric or a custom callable function. Supported
        string identifiers for predefined metrics are:
        - "cosine": Cosine similarity.
        - "abs_cosine": Absolute cosine similarity.
        - "correlation": Pearson correlation coefficient.
        - "euclidean": Inverse Euclidean distance (scaled to range 0-1).
        - "manhattan": Inverse Manhattan distance (scaled to range 0-1).
        - "chebyshev": Inverse Chebyshev distance (scaled to range 0-1).

        If a callable is provided, it must accept two NumPy arrays as input and
        return an array of pairwise similarity/distance scores.

    Returns:
    -------
    callable
        A function implementing the specified distance metric.

    Raises:
    -------
    ValueError:
        If the provided `distance` is not a recognized string identifier or a valid callable.

    Example:
    -------
    >>> distance_fn = get_distance_fn("cosine")
    >>> similarity_scores = distance_fn(x_sample, y_sample)
    """

    # Dictionary of supported distance metrics
    distance_metrics = {
        "abs_cosine": pairwise_abs_cosine,
        "cosine": pairwise_cosine,
        "correlation": pairwise_corr,
        "euclidean": pairwise_euclidean,
        "manhattan": pairwise_manhattan,
        "chebyshev": pairwise_chebyshev,
    }

    # If a string is provided, look up the corresponding metric function
    if isinstance(distance, str):
        if distance not in distance_metrics:
            raise ValueError(
                f"Unsupported distance metric: {distance}. Supported metrics are: {list(distance_metrics.keys())}"
            )
        distance_fn = distance_metrics[distance]
    elif callable(distance):
        # If a callable is provided, use it directly
        distance_fn = distance
    else:
        # Raise an error if neither a string nor a callable is provided
        raise ValueError("Distance must be either a string or a callable object.")

    # Wrap the distance function for efficient batch processing
    return batch_processing(distance_fn)


def random_binary_matrix(n, m, k, rng):
    """Generate a random binary matrix with a fixed number of 1's per row.

    This function creates an `n x m` binary matrix where each row contains exactly
    `k` ones, with the positions of the ones randomized using a specified random
    number generator (RNG).

    Parameters:
    ----------
    n : int
        Number of rows in the matrix.
    m : int
        Number of columns in the matrix.
    k : int
        Number of 1's to be placed in each row. Must satisfy `k <= m`.
    rng : np.random.Generator
        A NumPy random number generator instance used for shuffling the positions
        of the ones in each row.

    Returns:
    -------
    np.ndarray
        A binary matrix of shape `(n, m)` with exactly `k` ones per row.
    """
    # Initialize the binary matrix with all zeros
    matrix = np.zeros((n, m), dtype=int)

    # Fill the first `k` elements of each row with ones
    matrix[:, :k] = 1

    # Randomly shuffle each row to distribute the ones across the columns
    rng.permuted(matrix, axis=1, out=matrix)

    return matrix


def average_precision(rel_k):
    """Compute the Average Precision (AP) for a binary list of relevance scores.

    Average Precision (AP) is a performance metric for ranking tasks, which calculates
    the weighted mean of precision values at the positions where relevant items occur
    in a sorted list. The relevance list should be binary (1 for relevant items, 0
    for non-relevant).

    Parameters:
    ----------
    rel_k : np.ndarray
        A 2D binary array where each row represents a ranked list of items, and each
        element indicates the relevance of the item (1 for relevant, 0 for non-relevant).

    Returns:
    -------
    np.ndarray
        A 1D array of Average Precision (AP) scores, one for each row in the input array.
    """

    # Cumulative sum of relevance scores along the row (True Positives at each rank)
    tp = np.cumsum(rel_k, axis=1)

    # Total number of relevant items (last value in cumulative sum per row)
    num_pos = tp[:, -1]

    # Rank positions (1-based index for each column)
    k = np.arange(1, rel_k.shape[1] + 1)

    # Precision at each rank
    pr_k = tp / k

    # Calculate AP: Weighted sum of precision values, normalized by total relevant items
    ap = (pr_k * rel_k).sum(axis=1) / num_pos

    return ap


def ap_contiguous(
    rel_k_list: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Average Precision (AP) scores from relevance labels.

    This function calculates Average Precision (AP) scores for each profile based on
    relevance labels and their associated counts. It also returns configurations
    indicating the number of positive and total pairs for each profile.

    Parameters:
    ----------
    rel_k_list : np.ndarray
        Array of relevance labels (1 for positive pairs, 0 for negative pairs), sorted
        by descending similarity within profiles.
    counts : np.ndarray
        Array indicating how many times each profile appears in the rank list.

    Returns:
    -------
    ap_scores : np.ndarray
        Array of Average Precision scores for each profile.
    null_confs : np.ndarray
        Array of configurations, where each row corresponds to:
        - Number of positive pairs (`num_pos`).
        - Total number of pairs (`counts`).
    """
    # Convert counts into cutoff indices to segment relevance labels
    cutoffs = to_cutoffs(counts)

    # Calculate the number of positive pairs for each profile
    num_pos = np.add.reduceat(rel_k_list, cutoffs)

    # Compute the cumulative shift for handling contiguous true positives
    shift = np.empty_like(num_pos)
    shift[0], shift[1:] = 0, num_pos[:-1]

    # Calculate cumulative true positives for each profile segment
    tp = rel_k_list.cumsum() - np.repeat(shift.cumsum(), counts)

    # Rank positions for each relevance label, adjusted by cutoff indices
    k = np.arange(1, len(rel_k_list) + 1) - np.repeat(cutoffs, counts)

    # Compute precision at each rank (precision = TP / rank)
    pr_k = tp / k

    # Calculate average precision scores for each profile
    ap_scores = np.add.reduceat(pr_k * rel_k_list, cutoffs) / num_pos

    # Generate configurations (number of positive and total pairs)
    null_confs = np.stack([num_pos, counts], axis=1)

    return ap_scores, null_confs


def random_ap(num_perm, num_pos, total, seed):
    """
    Generate random Average Precision (AP) scores to create a null distribution.

    This function computes multiple Average Precision (AP) scores based on randomly
    generated binary relevance lists. It is useful for generating a null distribution
    to assess the significance of observed AP scores.

    Parameters:
    ----------
    num_perm : int
        Number of random permutations (i.e., how many random relevance lists to generate).
    num_pos : int
        Number of positive samples (1's) in each relevance list.
    total : int
        Total number of samples (columns) in each relevance list.
    seed : int
        Seed for the random number generator to ensure reproducibility.

    Returns:
    -------
    np.ndarray
        A 1D array containing the Average Precision scores for each randomly
        generated relevance list.
    """
    # Initialize the random number generator
    rng = np.random.default_rng(seed)

    # Generate a binary matrix with `num_perm` rows and `total` columns,
    # where each row contains exactly `num_pos` ones distributed randomly
    rel_k = random_binary_matrix(num_perm, total, num_pos, rng)

    # Compute Average Precision (AP) scores for each row of the binary matrix
    null_dist = average_precision(rel_k)

    return null_dist


def null_dist_cached(
    num_pos: int, total: int, seed: int, null_size: int, cache_dir: Path
) -> np.ndarray:
    """Generate or retrieve a cached null distribution for a given configuration.

    This function calculates a null distribution for a specified number of positive
    pairs (`num_pos`) and total pairs (`total`). It uses caching to store and
    retrieve precomputed distributions, saving time and computational resources.

    Parameters:
    ----------
    num_pos : int
        Number of positive pairs in the configuration.
    total : int
        Total number of pairs (positive + negative) in the configuration.
    seed : int
        Random seed for reproducibility.
    null_size : int
        Number of samples to generate in the null distribution.
    cache_dir : Path
        Directory to store or retrieve cached null distributions.

    Returns:
    -------
    np.ndarray
        Null distribution for the specified configuration.
    """
    # Check if a seed is provided to enable caching
    if seed is not None:
        # Define the cache file name based on the configuration
        cache_file = cache_dir / f"n{total}_k{num_pos}.npy"

        # If the cache file exists, load the null distribution from it
        if cache_file.is_file():
            null_dist = np.load(cache_file)
        else:
            # If the cache file doesn't exist, compute the null distribution
            null_dist = random_ap(null_size, num_pos, total, seed)

            # Save the computed distribution to the cache
            np.save(cache_file, null_dist)
    else:
        # If no seed is provided, compute the null distribution without caching
        null_dist = random_ap(null_size, num_pos, total, seed)

    # Return the null distribution (loaded or computed)
    return null_dist


def get_null_dists(confs: np.ndarray, null_size: int, seed: int) -> np.ndarray:
    """Generate null distributions for each configuration of positive and total pairs.
    Parameters:
    ----------
    confs : np.ndarray
        Array where each row contains the number of positive pairs (`num_pos`)
        and total pairs (`total`) for a specific configuration.
    null_size : int
        Number of samples to generate in the null distribution.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        A 2D array where each row corresponds to a null distribution for a specific
        configuration.
    """
    # Define the directory for caching null distributions
    cache_dir = Path.home() / ".copairs" / f"seed{seed}" / f"ns{null_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Number of configurations and random seeds for each configuration
    num_confs = len(confs)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(8096, size=num_confs)

    # Initialize an array to store null distributions
    null_dists = np.empty([len(confs), null_size], dtype=np.float32)

    # Function to generate null distributions for each configuration
    def par_func(i):
        num_pos, total = confs[i]
        null_dists[i] = null_dist_cached(num_pos, total, seeds[i], null_size, cache_dir)

    # Parallelize the generation of null distributions
    parallel_map(par_func, np.arange(num_confs))

    return null_dists


def p_values(ap_scores: np.ndarray, null_confs: np.ndarray, null_size: int, seed: int):
    """Calculate p-values for an array of Average Precision (AP) scores
    using a null distribution.

    Parameters:
    ----------
    ap_scores : np.ndarray
        Array of observed AP scores for which to calculate p-values.
    null_confs : np.ndarray
        Configuration array indicating the relevance or context of each AP score. Used
        to generate corresponding null distributions.
    null_size : int
        Number of samples to generate in the null distribution for each configuration.
    seed : int
        Seed for the random number generator to ensure reproducibility of the null
        distribution.

    Returns:
    -------
    np.ndarray
        An array of p-values corresponding to the input AP scores.
    """
    # Identify unique configurations and their indices
    confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)

    # Generate null distributions for each unique configuration
    null_dists = get_null_dists(confs, null_size, seed)

    # Sort null distributions for efficient p-value computation
    null_dists.sort(axis=1)

    # Initialize an array to store the p-values
    pvals = np.empty(len(ap_scores), dtype=np.float32)

    # Compute p-values for each AP score
    for i, (ap_score, ix) in enumerate(zip(ap_scores, rev_ix)):
        # Find the rank of the observed AP score in the sorted null distribution
        num = null_size - np.searchsorted(null_dists[ix], ap_score)

        # Calculate the p-value as the proportion of null scores >= observed score
        pvals[i] = (num + 1) / (null_size + 1)

    return pvals


def concat_ranges(start, end):
    """
    Create a 1D array by concatenating multiple integer ranges.

    This function generates a single concatenated array from multiple ranges defined
    by the `start` and `end` arrays. Each range is inclusive of `start` and exclusive
    of `end`.

    Parameters:
    ----------
    start : np.ndarray
        A 1D array of start indices for the ranges.
    end : np.ndarray
        A 1D array of end indices for the ranges. Must have the same shape as `start`.

    Returns:
    -------
    np.ndarray
        A 1D array containing the concatenated ranges.
    """
    # Generate individual ranges using `range` for each pair of start and end
    slices = map(range, start, end)

    # Flatten the ranges into a single iterable
    slices = itertools.chain.from_iterable(slices)

    # Calculate the total length of the concatenated ranges
    count = (end - start).sum()

    # Create a 1D array from the concatenated ranges
    mask = np.fromiter(slices, dtype=np.int32, count=count)

    return mask


def to_cutoffs(counts: np.ndarray) -> np.ndarray:
    """Convert counts into cumulative cutoff indices.

    This function generates a 1D array of indices that mark the start of each segment
    in a cumulative list. The first index is always `0`, and subsequent indices
    correspond to the cumulative sum of counts up to the previous entry.

    Parameters:
    ----------
    counts : np.ndarray
        A 1D array of counts representing the size of each segment.

    Returns:
    -------
    np.ndarray
        A 1D array of cutoff indices where each value indicates the starting index
        for the corresponding segment.
    """
    # Initialize an empty array for cutoff indices
    cutoffs = np.empty_like(counts)

    # Set the first cutoff to 0 (start of the first segment)
    cutoffs[0] = 0

    # Compute subsequent cutoffs using cumulative sums, excluding the last element
    cutoffs[1:] = counts.cumsum()[:-1]

    return cutoffs
