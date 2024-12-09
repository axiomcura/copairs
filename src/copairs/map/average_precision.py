import itertools
import logging

import numpy as np
import pandas as pd

from copairs import compute
from copairs.matching import Matcher, UnpairedException

from .filter import evaluate_and_filter, flatten_str_list, validate_pipeline_input

logger = logging.getLogger("copairs")


def build_rank_lists(pos_pairs, neg_pairs, pos_sims, neg_sims):
    labels = np.concatenate(
        [
            np.ones(pos_pairs.size, dtype=np.int32),
            np.zeros(neg_pairs.size, dtype=np.int32),
        ]
    )
    ix = np.concatenate([pos_pairs.ravel(), neg_pairs.ravel()])
    sim_all = np.concatenate([np.repeat(pos_sims, 2), np.repeat(neg_sims, 2)])
    ix_sort = np.lexsort([1 - sim_all, ix])
    rel_k_list = labels[ix_sort]
    paired_ix, counts = np.unique(ix, return_counts=True)
    return paired_ix, rel_k_list, counts


def average_precision(
    meta,
    feats,
    pos_sameby,
    pos_diffby,
    neg_sameby,
    neg_diffby,
    batch_size=20000,
    distance="cosine",
) -> pd.DataFrame:
    """
    Calculate average precision (AP) scores for pairs of profiles based on their
    similarity.

    This function identifies positive and negative pairs of profiles using  metadata
    rules, computes their similarity scores, and calculates average precision
    scores for each profile. The results include the number of positive and total pairs
    for each profile.

    Parameters:
    ----------
    meta : pd.DataFrame
        Metadata of the profiles, including columns used for defining pairs.
        This DataFrame should include the columns specified in `pos_sameby`, 
        `pos_diffby`, `neg_sameby`, and `neg_diffby`.

    feats : np.ndarray
        Feature matrix representing the profiles, where rows correspond to profiles 
        and columns to features.

    pos_sameby : list
        Metadata columns used to define positive pairs. Two profiles are considered a
        positive pair if they belong to the same group that is not a control group.
        For example, replicate profiles of the same compound are positive pairs and
        should share the same value in a column identifying compounds.

    pos_diffby : list
        Metadata columns used to differentiate positive pairs. Positive pairs do not need
        to differ in any metadata columns, so this is typically left empty. However,
        if necessary (e.g., to account for batch effects), you can specify columns
        such as batch identifiers.

    neg_sameby : list
        Metadata columns used to define negative pairs. Typically left empty, as profiles
        forming a negative pair (e.g., a compound and a DMSO/control) do not need to
        share any metadata values. This ensures comparisons are made without enforcing
        unnecessary constraints.

    neg_diffby : list
        Metadata columns used to differentiate negative pairs. Two profiles are considered
        a negative pair if one belongs to a compound group and the other to a DMSO/
        control group. They must differ in specified metadata columns, such as those
        identifying the compound and the treatment index, to ensure comparisons are
        only made between compounds and DMSO controls (not between different compounds).

    batch_size : int, optional
        The batch size for similarity computations to optimize memory usage.
        Default is 20,000.

    distance : str, optional
        The distance metric used for computing similarities. Default is "cosine".

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'average_precision': The calculated average precision score for each profile.
        - 'n_pos_pairs': The number of positive pairs for each profile.
        - 'n_total_pairs': The total number of pairs for each profile.
        - Additional metadata columns from the input.

    Raises:
    ------
    UnpairedException
        If no positive or negative pairs are found in the dataset.

    Notes:
    ------
    - Positive Pair Rules:
        * Positive pairs are defined by `pos_sameby` (profiles share these metadata values)
          and optionally differentiated by `pos_diffby` (profiles must differ in these metadata values if specified).
    - Negative Pair Rules:
        * Negative pairs are defined by `neg_diffby` (profiles differ in these metadata values)
          and optionally constrained by `neg_sameby` (profiles share these metadata values if specified).
    """

    # Combine all metadata columns needed for pair definitions
    columns = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)

    # Validate and filter metadata to ensure the required columns are present and usable
    meta, columns = evaluate_and_filter(meta, columns)
    validate_pipeline_input(meta, feats, columns)

    # Get the distance function for similarity calculations (e.g., cosine)
    distance_fn = compute.get_distance_fn(distance)

    # Reset metadata index for consistent indexing
    meta = meta.reset_index(drop=True).copy()

    # Initialize the Matcher object to find pairs based on metadata rules
    logger.info("Indexing metadata...")
    matcher = Matcher(meta, columns, seed=0)

    # Identify positive pairs based on `pos_sameby` and `pos_diffby`
    logger.info("Finding positive pairs...")
    pos_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_total = sum(len(p) for p in pos_pairs.values())
    if pos_total == 0:
        raise UnpairedException("Unable to find positive pairs.")

    # Convert positive pairs to a NumPy array for efficient computation
    pos_pairs = np.fromiter(
        itertools.chain.from_iterable(pos_pairs.values()),
        dtype=np.dtype((np.int32, 2)),
        count=pos_total,
    )

    # Identify negative pairs based on `neg_sameby` and `neg_diffby`
    logger.info("Finding negative pairs...")
    neg_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    neg_total = sum(len(p) for p in neg_pairs.values())
    if neg_total == 0:
        raise UnpairedException("Unable to find negative pairs.")

    # Convert negative pairs to a NumPy array for efficient computation
    neg_pairs = np.fromiter(
        itertools.chain.from_iterable(neg_pairs.values()),
        dtype=np.dtype((np.int32, 2)),
        count=neg_total,
    )

    # Compute similarities for positive pairs
    logger.info("Computing positive similarities...")
    pos_sims = distance_fn(feats, pos_pairs, batch_size)

    # Compute similarities for negative pairs
    logger.info("Computing negative similarities...")
    neg_sims = distance_fn(feats, neg_pairs, batch_size)

    # Build rank lists for calculating average precision
    logger.info("Building rank lists...")
    paired_ix, rel_k_list, counts = build_rank_lists(
        pos_pairs, neg_pairs, pos_sims, neg_sims
    )

    # Compute average precision scores and associated configurations
    logger.info("Computing average precision...")
    ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)

    # Add AP scores and pair counts to the metadata DataFrame
    logger.info("Creating result DataFrame...")
    meta["n_pos_pairs"] = 0
    meta["n_total_pairs"] = 0
    meta.loc[paired_ix, "average_precision"] = ap_scores
    meta.loc[paired_ix, "n_pos_pairs"] = null_confs[:, 0]
    meta.loc[paired_ix, "n_total_pairs"] = null_confs[:, 1]

    logger.info("Finished.")
    return meta


def p_values(dframe: pd.DataFrame, null_size: int, seed: int):
    """Compute p-values"""
    mask = dframe["n_pos_pairs"] > 0
    pvals = np.full(len(dframe), np.nan, dtype=np.float32)
    scores = dframe.loc[mask, "average_precision"].values
    null_confs = dframe.loc[mask, ["n_pos_pairs", "n_total_pairs"]].values
    pvals[mask] = compute.p_values(scores, null_confs, null_size, seed)
    return pvals
