import pandas as pd
from typing import Dict, Tuple, Any

def sample_with_counts(
    df: pd.DataFrame,
    group_label_counts: Dict[Tuple, int],
    random_state: int = None
) -> pd.DataFrame:
    """
    Uniformly sample from disjoint groups-label combinations with exact counts (without replacement).
    
    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    group_label_counts : dict
        Mapping from ((col1, val1), (col2, val2), ...) to desired count.
    random_state : int, optional
        Seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Sampled dataframe with specified counts per group.
    """
    samples = []
    
    for group_label_spec, count in group_label_counts.items():
        mask = pd.Series(True, index=df.index)
        for col, val in group_label_spec:
            mask &= (df[col] == val)
        
        group_label_df = df[mask]
        
        if len(group_label_df) < count:
            raise ValueError(
                f"Group {group_label_spec}: requested {count}, available {len(group_label_df)}"
            )
        
        samples.append(group_label_df.sample(n=count, random_state=random_state))
    
    return pd.concat(samples, ignore_index=True)


def max_initial_fraction(
    group_label_counts: Dict[Tuple, int],
    additions_by_params: Dict[Any, Dict[Tuple, int]]
) -> float:
    """
    Compute the maximum initial sample fraction such that the pool
    can satisfy additions for all parameter configurations.
    
    Parameters
    ----------
    group_label_counts : dict
        Mapping from group-label spec to total count n_{s,y}.
        These quantities are given by the dataset sketch.
    additions_by_params : dict
        Mapping from parameter configuration to {group_label: additions} dict.
        Only additions matter; deletions can always be satisfied from the initial sample.
    
    Returns
    -------
    float
        Maximum fraction x in (0, 1].
    """
    x_max = 1.0
    
    for additions in additions_by_params.values():
        for group, a in additions.items():
            if a > 0:
                n = group_label_counts[group]
                x_max = min(x_max, n / (n + a))
    
    return x_max