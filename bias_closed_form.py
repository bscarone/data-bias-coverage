"""
Analytical computation of bias metrics.

This module provides closed-form implementations for computing Uniform Bias from data or sketch representations, without invoking optimization solvers.

For bias *mitigation* via constrained optimization, see bias_optimization.py. 
"""
import pandas as pd
from itertools import combinations, product
from typing import Any
from dataframe_sketches import (get_total_count,
                                get_label_count,
                                get_group_count,
                                get_group_label_count,
                                get_attr_domain)

def uniform_bias(n:int, y:int, s:int, sy:int)->float:
    assert (n>=y and n>=s and y>=sy and s>=sy)
    fy = y/n
    fsy = sy/s
    return 1-(fsy/fy)

def compute_all_biases(df_sketch: pd.DataFrame, sensitive_attrs: list[str],
                       label_attr: str, include_marginals: bool = True) -> pd.DataFrame:
    """Compute Uniform Bias for intersectional and marginal groups."""
    n = get_total_count(df_sketch)
    label_domain = get_attr_domain(df_sketch, label_attr)
    
    records = []
    
    # Iterate over all non-empty subsets of sensitive attributes
    min_size = 1 if include_marginals else len(sensitive_attrs)
    for r in range(min_size, len(sensitive_attrs) + 1):
        for attr_subset in combinations(sensitive_attrs, r):
            attr_domains = [get_attr_domain(df_sketch, attr) 
                            for attr in attr_subset]
            
            for group_vals in product(*attr_domains):
                group = list(zip(attr_subset, group_vals))
                
                for label_val in label_domain:
                    label = (label_attr, label_val)
                    
                    y = get_label_count(df_sketch, label)
                    s = get_group_count(df_sketch, group)
                    sy = get_group_label_count(df_sketch, group, label)
                    
                    if s == 0 or y == 0:
                        continue
                    
                    f_sy = sy / s
                    f_y = y / n
                    b = uniform_bias(n, y, s, sy)
                    
                    # Build full-length tuple with '*' for marginalized attrs
                    group_tuple = tuple(
                        dict(group).get(attr, '*') for attr in sensitive_attrs
                    )
                    
                    records.append({
                        'group': group_tuple,
                        'label': label_val,
                        'bias': b,
                        'f_sy': f_sy,
                        'f_y': f_y,
                        '|f_sy-f_y|': abs(f_sy - f_y)
                    })
    
    return pd.DataFrame(records)


def get_max_bias(
    csv_path: str,
    column: str = "bias",
    intersectional_only: bool = False
) -> tuple[float, str, str]:
    """
    Return the max value of the selected column from a bias CSV file,
    along with its group and label.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing bias data.
    column : str
        Column name to compute the max over (default: "bias").
    intersectional_only : bool
        If True, only consider complete/intersectional groups (no '*' in group).
        If False, consider all entries (default: False).
    
    Returns
    -------
    tuple[float, str, str]
        (max_value, group, label)
    """
    df = pd.read_csv(csv_path)
    print(repr(df.columns.tolist())) # [debug]
    
    if intersectional_only:
        # Filter out rows where the group contains '*'
        df = df[~df["group"].str.contains(r"\*", regex=True)]
    
    idx = df[column].idxmax()
    row = df.loc[idx]
    
    return float(row[column]), row["group"], row["label"]


def get_group_free_variable_ith_label(
    df_sketch: pd.DataFrame,
    group: list[tuple[str, Any]],
    label_attr: str,
    i: int = -1
) -> Any:
    """
    Return the label with the i-th smallest sy/y ratio for a fixed group.
    
    The label acts as the free variable; the group is held constant.
    Use i=-1 for argmax, i=0 for argmin.
    """
    label_domain = get_attr_domain(df_sketch, label_attr)
    n = len(label_domain)
    assert -n <= i < n, f"i={i} out of range for {n} labels"
    
    candidates = {}
    for label_val in label_domain:
        label = (label_attr, label_val)
        y = get_label_count(df_sketch, label)
        sy = get_group_label_count(df_sketch, group, label)
        
        if y == 0:
            continue
        
        candidates[label_val] = sy / y
    
    sorted_lst = sorted(candidates.items(), key=lambda x: x[1])
    return sorted_lst[i][0]