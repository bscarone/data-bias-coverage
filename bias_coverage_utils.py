"""
Utility functions for bias-coverage experiments.
"""
import itertools
from typing import Any

import pandas as pd

# Type aliases
Group = tuple[tuple[str, str], ...]
GroupLabel = tuple[tuple[str, str], ...]


def get_groups(
    df: pd.DataFrame,
    sensitive_attrs: list[str]
) -> list[Group]:
    """
    Generate all intersectional groups from a dataset.
    
    Args:
        df: DataFrame containing the data
        sensitive_attrs: List of column names for sensitive attributes
        
    Returns:
        List of groups, where each group is a tuple of (attr, value) pairs
        
    Example:
        >>> df = pd.DataFrame({'sex': ['M', 'F', 'M'], 'race': ['W', 'B', 'W']})
        >>> get_groups(df, ['sex', 'race'])
        [
            (('sex', 'M'), ('race', 'W')),
            (('sex', 'M'), ('race', 'B')),
            (('sex', 'F'), ('race', 'W')),
            (('sex', 'F'), ('race', 'B')),
        ]
    """
    attr_values = [list(df[attr].unique()) for attr in sensitive_attrs]
    
    groups = [
        tuple((attr, val) for attr, val in zip(sensitive_attrs, combo))
        for combo in itertools.product(*attr_values)
    ]
    
    return groups


def get_grouplabels(
    df: pd.DataFrame,
    sensitive_attrs: list[str],
    label_attr: str
) -> list[GroupLabel]:
    """
    Generate all group-label combinations.
    
    Args:
        df: DataFrame containing the data
        sensitive_attrs: List of column names for sensitive attributes
        label_attr: Column name for the label/outcome
        
    Returns:
        List of group-labels, each a tuple of (attr, value) pairs
        including the label as the last element
    """
    groups = get_groups(df, sensitive_attrs)
    label_vals = list(df[label_attr].unique())
    
    return [
        tuple(list(group) + [(label_attr, lv)])
        for group in groups
        for lv in label_vals
    ]


def get_group_counts(
    df: pd.DataFrame,
    sensitive_attrs: list[str],
    label_attr: str
) -> dict[GroupLabel, int]:
    """
    Count occurrences of each group-label combination.
    
    Args:
        df: DataFrame containing the data
        sensitive_attrs: List of column names for sensitive attributes
        label_attr: Column name for the label/outcome
        
    Returns:
        Dictionary mapping group-labels to their counts (sy values)
    """
    grouplabels = get_grouplabels(df, sensitive_attrs, label_attr)
    
    counts = {}
    for gl in grouplabels:
        mask = pd.Series(True, index=df.index)
        for attr, val in gl:
            # &= is element-wise and
            mask &= (df[attr] == val)
        counts[gl] = mask.sum()
    
    return counts


def get_label_fractions(
    df: pd.DataFrame,
    label_attr: str
) -> dict[str, float]:
    """
    Compute f_y = |y| / n for each label value.
    
    Args:
        df: DataFrame containing the data
        label_attr: Column name for the label/outcome
        
    Returns:
        Dictionary mapping label values to their fractions
    """
    n = len(df)
    return {
        val: (df[label_attr] == val).sum() / n
        for val in df[label_attr].unique()
    }


def get_default_coverage(
    df: pd.DataFrame,
    sensitive_attrs: list[str],
    label_attr: str,
    min_count: int = 1
) -> dict[GroupLabel, int]:
    """
    Generate default coverage requirements.
    
    Sets m_sy = max(min_count, current count), ensuring
    no group-label loses representation.
    
    Args:
        df: DataFrame containing the data
        sensitive_attrs: List of column names for sensitive attributes
        label_attr: Column name for the label/outcome
        min_count: Minimum coverage for any group-label
        
    Returns:
        Dictionary mapping group-labels to minimum coverage
    """
    counts = get_group_counts(df, sensitive_attrs, label_attr)
    return {gl: max(min_count, count) for gl, count in counts.items()}


def get_scaled_coverage(
    df: pd.DataFrame,
    sensitive_attrs: list[str],
    label_attr: str,
    scale: float = 1.0,
    min_count: int = 1
) -> dict[GroupLabel, int]:
    """
    Generate scaled coverage requirements.
    
    Sets m_sy = max(min_count, scale * current count).
    
    Args:
        df: DataFrame containing the data
        sensitive_attrs: List of column names for sensitive attributes
        label_attr: Column name for the label/outcome
        scale: Multiplier for current counts
        min_count: Minimum coverage for any group-label
    """
    counts = get_group_counts(df, sensitive_attrs, label_attr)
    return {
        gl: max(min_count, int(scale * count)) 
        for gl, count in counts.items()
    }


def prepare_ilp_inputs(
    df: pd.DataFrame,
    sensitive_attrs: list[str],
    label_attr: str,
    coverage_scale: float = 1.0,
    min_coverage: int = 1
) -> dict[str, Any]:
    """
    Prepare all inputs needed for bias_coverage_mitigation_ilp.
    
    Convenience function that computes groups, counts, fractions,
    and coverage requirements from a raw DataFrame.
    
    Args:
        df: DataFrame containing the data
        sensitive_attrs: List of column names for sensitive attributes
        label_attr: Column name for the label/outcome
        coverage_scale: Multiplier for coverage requirements
        min_coverage: Minimum coverage for any group-label
        
    Returns:
        Dictionary with keys: groups, grouplabel_coverage, original_sy, f_y
        
    Example:
        >>> inputs = prepare_ilp_inputs(df, ['sex', 'race'], 'outcome')
        >>> solution = bias_coverage_mitigation_ilp(
        ...     df_sketch, label_attr='outcome',
        ...     groups=inputs['groups'],
        ...     grouplabel_coverage=inputs['grouplabel_coverage'],
        ...     eps=0.1
        ... )
    """
    return {
        'groups': get_groups(df, sensitive_attrs),
        'grouplabel_coverage': get_scaled_coverage(
            df, sensitive_attrs, label_attr, coverage_scale, min_coverage
        ),
        'original_sy': get_group_counts(df, sensitive_attrs, label_attr),
        'f_y': get_label_fractions(df, label_attr),
    }

