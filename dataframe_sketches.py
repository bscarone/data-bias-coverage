"""
Sketch operations for bias computation.

This module provides functions for querying contingency table sketches,
which aggregate tuple counts by attribute values.

Sketch structure:
    A sketch is a pandas DataFrame where each row represents a unique
    combination of attribute values, with a 'count' column storing the
    number of tuples with that combination. For example:

        sex      race     class  count
        Female   Black    0      42
        Female   Black    1      18
        Female   White    0      105
        ...

    This is equivalent to the result of:
        df.groupby([*sensitive_attrs, label_attr]).size().reset_index(name='count')

Notation (following the Uniform Bias formulation):
    n   - total number of tuples in the dataset
    y   - number of tuples with a specific label value
    s   - number of tuples belonging to a demographic group 
          (defined by sensitive attribute values)
    sy  - number of tuples in the intersection of group s with label y

A "group" is specified as a list of (attribute, value) pairs representing
a conjunction of equality conditions, e.g., [("sex", "Female"), ("race", "Black")].
"""
# Standard library imports
from __future__ import annotations
from typing import Any

# Third-party imports
import pandas as pd

def generate_sketch(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    count_col: str = "count",
    sort: bool = True
) -> pd.DataFrame:
    """
    Generate a sketch (contingency table) from a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list[str], optional
        Columns to include in the sketch. If None, uses all columns.
    count_col : str
        Name for the count column in the output.
    sort : bool
        Whether to sort by the grouping columns.
    
    Returns
    -------
    pd.DataFrame
        Sketch with unique value combinations and their counts.
    """
    cols = df.columns.tolist()
    sketch = df.groupby(cols, dropna=False).size().reset_index(name=count_col)
    
    # Warn if NAs are present
    if df.isna().any().any():
        na_counts = df.isna().sum()
        cols_with_na = na_counts[na_counts > 0].to_dict()
        print(f"Warning: NA values found in columns: {cols_with_na}")
    
    return sketch


def dict_to_dataframe(data_dict):
    rows = []
    for key, value in data_dict.items():
        # Extract the values from the nested tuple structure
        row = {field: field_value for field, field_value in key}
        row['count'] = value
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_grouplabel_dict(df: pd.DataFrame, count_col: str = 'count') -> dict:
    """Convert a DataFrame to a GroupLabel dict."""
    result = {}
    attr_cols = [c for c in df.columns if c != count_col]
    
    for _, row in df.iterrows():
        key = tuple((col, row[col]) for col in attr_cols)
        result[key] = row[count_col]
    
    return result


def sum_grouplabel_dfs(df1: pd.DataFrame, df2: pd.DataFrame, 
                       count_col: str = 'count') -> pd.DataFrame:
    """Sum counts of two GroupLabel DataFrames."""
    attr_cols = [c for c in df1.columns if c != count_col]
    
    merged = df1.merge(df2, on=attr_cols, how='outer', suffixes=('_1', '_2'))
    merged[f'{count_col}_1'] = merged[f'{count_col}_1'].fillna(0)
    merged[f'{count_col}_2'] = merged[f'{count_col}_2'].fillna(0)
    merged[count_col] = merged[f'{count_col}_1'] + merged[f'{count_col}_2']
    
    return merged[attr_cols + [count_col]]

def add_dicts(dict1:dict, dict2:dict):
    assert dict1.keys() == dict2.keys()
    return {key: dict1[key] + dict2[key] for key in dict1}

def select_rows_by_values(df: pd.DataFrame,
                          attrs_values: list[tuple[str, Any]]) -> pd.DataFrame:
    """Select rows where each attribute equals its specified value (conjunction)."""
    if not attrs_values:
        return df
    mask = pd.Series(True, index=df.index)
    for col, val in attrs_values:
        mask &= (df[col] == val)
    return df.loc[mask]


def get_attr_domain(df_sketch: pd.DataFrame, attr: str) -> list:
    """Return the domain of a single attribute from the sketch."""
    return df_sketch[attr].dropna().unique().tolist()


def get_attr_domains(df_sketch: pd.DataFrame, sensitive_attrs: list[str], 
                     label_attr: str) -> list[list]:
    """Return domains for sensitive attributes + label attribute."""
    return [get_attr_domain(df_sketch, attr) 
            for attr in [*sensitive_attrs, label_attr]]
    

def get_total_count(sketch: pd.DataFrame) -> int:
    """Count all tuples in the dataset (n)."""
    return sketch['count'].sum()


def get_label_count(sketch: pd.DataFrame, label: tuple[str, Any]) -> int:
    """Count tuples with a specific label value (y)."""
    return get_group_count(sketch, [label])


def get_group_count(sketch: pd.DataFrame,
                    group: list[tuple[str, Any]]) -> int:
    """Count tuples belonging to a demographic group (s)."""
    return select_rows_by_values(sketch, group)['count'].sum()


def get_group_label_count(sketch: pd.DataFrame,
                          group: list[tuple[str, Any]], 
                          label: tuple[str, Any]) -> int:
    """Count tuples with given group membership AND label value (sy)."""
    return get_group_count(sketch, group + [label])


def get_label_rate(sketch: pd.DataFrame, label: tuple[str, Any]) -> float:
    """Fraction of tuples with a specific label value (y/n).
    
    Returns NaN if the sketch is empty.
    """
    y = get_label_count(sketch, label)
    n = get_total_count(sketch)
    if n == 0:
        return float('nan')
    return y / n


def get_group_label_rate(sketch: pd.DataFrame,
                         group: list[tuple[str, Any]],
                         label: tuple[str, Any]) -> float:
    """Fraction of group members with a specific label value (sy/s).
    
    Returns NaN if the group is empty (no representation in data).
    """
    sy = get_group_label_count(sketch, group, label)
    s = get_group_count(sketch, group)
    if s == 0:
        return float('nan')
    return sy / s