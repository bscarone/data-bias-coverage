"""
Closed-form solution for bias mitigation.

This module computes the analytical solution for Δ[sy] that achieves 
zero Uniform Bias, without invoking optimization solvers.

For ILP-based solutions, see bias_ilp_solution.py
"""
from __future__ import annotations

import math
import pandas as pd
from itertools import combinations, product
from typing import Any

from dataframe_sketches import (
    get_label_count,
    get_group_count,
    get_group_label_count,
    get_attr_domain
)
from bias_closed_form import get_group_free_variable_ith_label, uniform_bias


def compute_group_exact_solution(
    df_sketch: pd.DataFrame,
    group: list[tuple[str, Any]],
    label_attr: str,
    m_sy: int = 1000,
    k: int | None = None
) -> dict[Any, float]:
    """
    Compute exact Δ[sy] = -[sy] + k*y achieving exactly zero bias.
    
    Parameters
    ----------
    k : int | None
        Multiplier for exact solution. If None, uses minimum k = ceil(m_sy / y)
        per label to satisfy coverage. If provided, must be >= ceil(m_sy / y).
    
    Returns
    -------
    dict[Any, float]
        Mapping from label value to Δ[sy]
    """
    label_domain = get_attr_domain(df_sketch, label_attr)
    
    solution = {}
    for label_val in label_domain:
        sy = get_group_label_count(df_sketch, group, (label_attr, label_val))
        y = get_label_count(df_sketch, (label_attr, label_val))
        k_min = math.ceil(m_sy / y)
        
        if k is None:
            k_used = k_min
        else:
            if k < k_min:
                raise ValueError(
                    f"k={k} too small for label {label_val}: need k >= {k_min} to satisfy coverage"
                )
            k_used = k
        
        delta_sy = -sy + k_used * y
        solution[label_val] = delta_sy
    
    return solution


def compute_delta_sy(
    df_sketch: pd.DataFrame,
    group: list[tuple[str, Any]],
    label_attr: str,
    y_i: Any,
    m_sy: int = 1000
) -> int:
    """
    Compute Δ[sy_i] = max_y { ⌈(y_i/y) * m_sy - sy_i⌉ }
    
    Parameters
    ----------
    y_i : Any
        The free variable label (from get_group_free_variable_ith_label)
    m_sy : int
        Target count for each group-label pair (default: 1000)
    """
    label_domain = get_attr_domain(df_sketch, label_attr)
    
    y_i_count = get_label_count(df_sketch, (label_attr, y_i))
    sy_i = get_group_label_count(df_sketch, group, (label_attr, y_i))
    
    max_delta = float('-inf')
    
    for label_val in label_domain:
        y = get_label_count(df_sketch, (label_attr, label_val))
        
        if y == 0:
            continue
        
        delta = math.ceil((y_i_count / y) * m_sy - sy_i)
        max_delta = max(max_delta, delta)
    
    return max_delta


def compute_group_approximate_solution(
    df_sketch: pd.DataFrame,
    group: list[tuple[str, Any]],
    label_attr: str,
    m_sy: int = 1000,
    rounding: str | None = None
) -> dict[Any, float | int]:
    """
    Compute Δ[sy] for all labels for a fixed group.
    
    Steps:
    1. Find y_i = argmax_y (sy/y)
    2. Compute Δ[sy_i] = max_y { ⌈(y_i/y) * m_sy - sy_i⌉ }
    3. For each label y: Δ[sy] = -sy + (y/y_i) * (sy_i + Δ[sy_i])
    
    Parameters
    ----------
    rounding : str | None
        Rounding mode: 'ceil', 'floor', 'round', or None for no rounding.
    
    Returns
    -------
    dict[Any, float | int]
        Mapping from label value to Δ[sy]
    """
    round_fn = {
        'ceil': math.ceil,
        'floor': math.floor,
        'round': round,
        None: lambda x: x
    }[rounding]
    
    # Step 1: find free variable
    y_i = get_group_free_variable_ith_label(df_sketch, group, label_attr, i=-1)
    
    # Step 2: compute delta for free variable
    delta_sy_i = compute_delta_sy(df_sketch, group, label_attr, y_i, m_sy)
    
    # Precompute counts for y_i
    y_i_count = get_label_count(df_sketch, (label_attr, y_i))
    sy_i = get_group_label_count(df_sketch, group, (label_attr, y_i))
    
    # Step 3: compute delta for all labels
    label_domain = get_attr_domain(df_sketch, label_attr)
    solution = {}
    
    for label_val in label_domain:
        y = get_label_count(df_sketch, (label_attr, label_val))
        sy = get_group_label_count(df_sketch, group, (label_attr, label_val))
        
        delta_sy = -sy + (y / y_i_count) * (sy_i + delta_sy_i)
        solution[label_val] = round_fn(delta_sy)
    
    return solution


def compute_all_group_exact_solutions(
    df_sketch: pd.DataFrame,
    sensitive_attrs: list[str],
    label_attr: str,
    m_sy: int = 1000,
    k: int | None = None,
    intersectional_only: bool = True,
    verify: bool = True,
    atol: float = 1e-3
) -> pd.DataFrame:
    """
    Compute exact Δ[sy] = -[sy] + k*y for all group-label pairs.
    
    This achieves exactly zero bias (not approximate).
    
    Parameters
    ----------
    k : int | None
        Multiplier for exact solution. If None, uses minimum k = ceil(m_sy / y)
        per label. If provided, must be >= ceil(m_sy / y) for all labels.
    intersectional_only : bool
        If True, only compute for full intersectional groups.
        If False, also include marginal groups.
    verify : bool
        If True, verify that new_sy >= m_sy and bias = 0.
    atol : float
        Absolute tolerance for verification (default: 1e-3).
    
    Returns
    -------
    pd.DataFrame
        Columns: group, label, sy, y, delta_sy, new_sy, f_y, new_f_sy, new_f_y, new_bias
    """
    records = []
    
    min_size = len(sensitive_attrs) if intersectional_only else 1
    
    for r in range(min_size, len(sensitive_attrs) + 1):
        for attr_subset in combinations(sensitive_attrs, r):
            attr_domains = [get_attr_domain(df_sketch, attr) for attr in attr_subset]
            
            for group_vals in product(*attr_domains):
                group = list(zip(attr_subset, group_vals))
                
                # Skip empty groups
                if get_group_count(df_sketch, group) == 0:
                    continue
                
                solution = compute_group_exact_solution(
                    df_sketch, group, label_attr, m_sy, k
                )
                
                group_tuple = tuple(
                    dict(group).get(attr, '*') for attr in sensitive_attrs
                )
                
                for label_val, delta_sy in solution.items():
                    label = (label_attr, label_val)
                    sy = get_group_label_count(df_sketch, group, label)
                    y = get_label_count(df_sketch, label)
                    records.append({
                        'group': group_tuple,
                        'label': label_val,
                        'sy': sy,
                        'y': y,
                        'delta_sy': delta_sy,
                        'new_sy': sy + delta_sy
                    })
    
    df = pd.DataFrame(records)
    
    # Compute original totals
    n = df.groupby('label').first()['y'].sum()
    df['f_y'] = df['y'] / n
    
    # Compute new totals
    new_n = df['new_sy'].sum()
    new_y = df.groupby('label')['new_sy'].transform('sum')
    new_s = df.groupby('group')['new_sy'].transform('sum')
    
    df['new_f_sy'] = df['new_sy'] / new_s
    df['new_f_y'] = new_y / new_n
    
    # Compute new bias
    new_y_dict = df.groupby('label')['new_sy'].sum().to_dict()
    new_s_dict = df.groupby('group')['new_sy'].sum().to_dict()
    
    df['new_bias'] = df.apply(
        lambda row: uniform_bias(
            int(new_n),
            int(new_y_dict[row['label']]),
            int(new_s_dict[row['group']]),
            int(row['new_sy'])
        ) if new_s_dict[row['group']] > 0 and new_y_dict[row['label']] > 0 else None,
        axis=1
    )
    
    avg_new_bias = df['new_bias'].abs().mean()
    total_delta = df['delta_sy'].sum()
    print(f"Average |new_bias|: {avg_new_bias:.6f}")
    print(f"Total Δ[sy]: {total_delta:.0f}")
    
    if verify:
        _verify_solution(df, m_sy, atol)
    
    return df


def compute_all_group_approximate_solutions(
    df_sketch: pd.DataFrame,
    sensitive_attrs: list[str],
    label_attr: str,
    m_sy: int = 1000,
    intersectional_only: bool = True,
    rounding: str | None = None,
    verify: bool = True,
    atol: float = 1e-3
) -> pd.DataFrame:
    """
    Compute Δ[sy] for all group-label pairs.
    
    Parameters
    ----------
    intersectional_only : bool
        If True, only compute for full intersectional groups.
        If False, also include marginal groups.
    rounding : str | None
        Rounding mode: 'ceil', 'floor', 'round', or None for no rounding.
    verify : bool
        If True, verify that new_sy ≈ m_sy for all rows.
    atol : float
        Absolute tolerance for verification (default: 1e-3).
    
    Returns
    -------
    pd.DataFrame
        Columns: group, label, sy, delta_sy, new_sy
    """
    records = []
    
    min_size = len(sensitive_attrs) if intersectional_only else 1
    
    for r in range(min_size, len(sensitive_attrs) + 1):
        for attr_subset in combinations(sensitive_attrs, r):
            attr_domains = [get_attr_domain(df_sketch, attr) for attr in attr_subset]
            
            for group_vals in product(*attr_domains):
                group = list(zip(attr_subset, group_vals))
                
                # Skip empty groups
                if get_group_count(df_sketch, group) == 0:
                    continue
                
                solution = compute_group_approximate_solution(
                    df_sketch, group, label_attr, m_sy, rounding
                )
                
                group_tuple = tuple(
                    dict(group).get(attr, '*') for attr in sensitive_attrs
                )
                
                for label_val, delta_sy in solution.items():
                    label = (label_attr, label_val)
                    sy = get_group_label_count(df_sketch, group, label)
                    y = get_label_count(df_sketch, label)
                    records.append({
                        'group': group_tuple,
                        'label': label_val,
                        'sy': sy,
                        'y': y,
                        'delta_sy': delta_sy,
                        'new_sy': sy + delta_sy
                    })
    
    df = pd.DataFrame(records)
    
    # Compute original totals
    n = df.groupby('label').first()['y'].sum()  # original n
    df['f_y'] = df['y'] / n
    
    # Compute new totals and add f_sy, f_y columns
    new_n = df['new_sy'].sum()
    new_y = df.groupby('label')['new_sy'].transform('sum')
    new_s = df.groupby('group')['new_sy'].transform('sum')
    
    df['new_f_sy'] = df['new_sy'] / new_s
    df['new_f_y'] = new_y / new_n
    
    # Compute new bias for each row
    new_y_dict = df.groupby('label')['new_sy'].sum().to_dict()
    new_s_dict = df.groupby('group')['new_sy'].sum().to_dict()
    
    df['new_bias'] = df.apply(
        lambda row: uniform_bias(
            int(new_n), 
            int(new_y_dict[row['label']]), 
            int(new_s_dict[row['group']]), 
            int(row['new_sy'])
        ) if new_s_dict[row['group']] > 0 and new_y_dict[row['label']] > 0 else None,
        axis=1
    )
    
    avg_new_bias = df['new_bias'].abs().mean()
    print(f"Average |new_bias|: {avg_new_bias:.6f}")
    
    if verify:
        _verify_solution(df, m_sy, atol)
    
    return df


def _verify_solution(
    df: pd.DataFrame,
    m_sy: int,
    atol: float
) -> None:
    """
    Verify the solution satisfies:
    1. Zero bias: uniform_bias(n, y, s, sy) ≈ 0 for all group-label pairs
    2. Coverage: new_sy >= m_sy
    3. No over-deletion: delta_sy >= -sy (i.e., new_sy >= 0)
    """
    errors = []
    
    # Check 3: No over-deletion
    over_deleted = df[df['new_sy'] < 0]
    if len(over_deleted) > 0:
        sample = over_deleted.head(3).to_string()
        errors.append(f"Over-deletion: {len(over_deleted)} rows have new_sy < 0.\nSample:\n{sample}")
    
    # Check 2: Coverage constraint
    under_coverage = df[df['new_sy'] < m_sy]
    if len(under_coverage) > 0:
        sample = under_coverage.head(3).to_string()
        errors.append(f"Coverage violation: {len(under_coverage)} rows have new_sy < m_sy={m_sy}.\nSample:\n{sample}")
    
    # Check 1: Zero bias
    # Compute new totals from modified counts
    new_n = df['new_sy'].sum()
    new_y = df.groupby('label')['new_sy'].sum().to_dict()
    new_s = df.groupby('group')['new_sy'].sum().to_dict()
    
    bias_violations = []
    for _, row in df.iterrows():
        n = new_n
        y = new_y[row['label']]
        s = new_s[row['group']]
        sy = row['new_sy']
        
        if s > 0 and y > 0:
            b = uniform_bias(n, y, s, sy)
            if abs(b) > atol:
                bias_violations.append({
                    'group': row['group'],
                    'label': row['label'],
                    'bias': b
                })
    
    if bias_violations:
        sample = pd.DataFrame(bias_violations).head(3).to_string()
        errors.append(f"Zero-bias violation: {len(bias_violations)} rows have |bias| > {atol}.\nSample:\n{sample}")
    
    if errors:
        raise ValueError("Verification failed:\n\n" + "\n\n".join(errors))