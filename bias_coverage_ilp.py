"""
Solves the bias-coverage problem with linearized fairness constraints.
Uses the formulation: |f_{sy}^{new} - f_y| <= epsilon
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from dataframe_sketches import (get_group_label_count,get_group_count,
                                get_label_count,get_total_count)
    

logger = logging.getLogger(__name__)

# Constants
VAR_LOWER_BOUND = -10000

# Type aliases
Group = tuple[tuple[str, str], ...]
GroupLabel = tuple[tuple[str, str], ...]


def group_key(group: Group) -> str:
    """Convert group tuple to string key for variable naming."""
    return ''.join(str(val) for _, val in group)


def grouplabel_key(group: Group, label_val: str) -> str:
    """Convert group-label pair to string key."""
    return group_key(group) + label_val


@dataclass
class BiasCoverageModel:
    """Encapsulates the Gurobi model and associated variables."""
    model: gp.Model
    # Decision variables
    delta_sy: dict[str, gp.Var]
    delta_pos: dict[str, gp.Var]  # Δ⁺_sy (additions)
    delta_neg: dict[str, gp.Var]  # Δ⁻_sy (deletions)
    delta_s: dict[str, gp.Var]
    # Fixed quantities (as variables for constraint building)
    sy: dict[str, int]  # Original counts (constants, not variables)
    y: dict[str, int]
    s: dict[str, int]
    n: int
    f_y: dict[str, float]  # f_y = y / n

    @classmethod
    def create(cls) -> "BiasCoverageModel":
        return cls(
            model=gp.Model("bias_coverage"),
            delta_sy={}, delta_pos={}, delta_neg={}, delta_s={},
            sy={}, y={}, s={}, n=0, f_y={}
        )


def add_group_label_variables(
    bcm: BiasCoverageModel,
    df_sketch: pd.DataFrame,
    groups: list[Group],
    label_vals: list[str],
    label_attr: str
) -> None:
    """Add Delta_sy with decomposition into Δ⁺ and Δ⁻."""
    m = bcm.model
    
    for group in groups:
        gk = group_key(group)
        for label_val in label_vals:
            glk = gk + str(label_val)
            
            # Δ⁺_sy >= 0 (additions)
            bcm.delta_pos[glk] = m.addVar(
                vtype=GRB.INTEGER, name=f"Delta_pos_{glk}", lb=0
            )
            # Δ⁻_sy >= 0 (deletions)
            bcm.delta_neg[glk] = m.addVar(
                vtype=GRB.INTEGER, name=f"Delta_neg_{glk}", lb=0
            )
            # Δ_sy = Δ⁺_sy - Δ⁻_sy (can be negative)
            bcm.delta_sy[glk] = m.addVar(
                vtype=GRB.INTEGER, name=f"Delta_{glk}", lb=VAR_LOWER_BOUND
            )
            m.addConstr(
                bcm.delta_sy[glk] == bcm.delta_pos[glk] - bcm.delta_neg[glk],
                name=f"decomp_{glk}"
            )
            
            # Store original count
            sy_val = get_group_label_count(df_sketch, list(group), 
                                           (label_attr, label_val))
            bcm.sy[glk] = sy_val
            
            # Can't delete more than we have
            m.addConstr(bcm.delta_neg[glk] <= sy_val, name=f"max_del_{glk}")


def add_label_marginals(
    bcm: BiasCoverageModel,
    df_sketch: pd.DataFrame,
    label_vals: list[str],
    label_attr: str
) -> None:
    """Store label marginals and compute f_y."""
    for label_val in label_vals:
        y_val = get_label_count(df_sketch, (label_attr, label_val))
        bcm.y[label_val] = y_val
        bcm.f_y[label_val] = y_val / bcm.n


def add_group_variables(
    bcm: BiasCoverageModel,
    df_sketch: pd.DataFrame,
    groups: list[Group],
    label_vals: list[str]
) -> None:
    """Add Delta_s variables with constraint: Δs = Σ_y Δsy."""
    m = bcm.model
    
    for group in groups:
        gk = group_key(group)
        
        bcm.delta_s[gk] = m.addVar(
            vtype=GRB.INTEGER, name=f"Delta_{gk}", lb=VAR_LOWER_BOUND
        )
        m.addConstr(
            bcm.delta_s[gk] == gp.quicksum(bcm.delta_sy[gk + str(lv)] for lv in label_vals),
            name=f"delta_s_def_{gk}"
        )
        
        # Store original group size
        bcm.s[gk] = get_group_count(df_sketch, group) # type: ignore


def add_fairness_constraints(
    bcm: BiasCoverageModel,
    groups: list[Group],
    label_vals: list[str],
    eps: float
) -> None:
    """
    Add linearized fairness constraints.
    
    Original: |f_{sy}^{new} - f_y| <= eps
    Where f_{sy}^{new} = (sy + Δsy) / (s + Δs)
    
    Linearized form (multiplying by denominator, assuming s + Δs > 0):
      Upper: Δsy - (f_y + ε)Δs <= (f_y + ε)s - sy
      Lower: Δsy - (f_y - ε)Δs >= (f_y - ε)s - sy
    """
    m = bcm.model
    
    for group in groups:
        gk = group_key(group)
        s = bcm.s[gk]
        
        for label_val in label_vals:
            glk = gk + str(label_val)
            sy = bcm.sy[glk]
            f_y = bcm.f_y[label_val]
            
            # Upper bound: f_sy^new <= f_y + eps
            # Δsy - (f_y + ε)Δs <= (f_y + ε)s - sy
            m.addConstr(
                bcm.delta_sy[glk] - (f_y + eps) * bcm.delta_s[gk] 
                <= (f_y + eps) * s - sy,
                name=f"fair_upper_{glk}"
            )
            
            # Lower bound: f_sy^new >= f_y - eps
            # Δsy - (f_y - ε)Δs >= (f_y - ε)s - sy
            m.addConstr(
                bcm.delta_sy[glk] - (f_y - eps) * bcm.delta_s[gk] 
                >= (f_y - eps) * s - sy,
                name=f"fair_lower_{glk}"
            )


def add_positive_group_size_constraints(
    bcm: BiasCoverageModel,
    groups: list[Group],
    min_group_size: int = 1
) -> None:
    """Ensure s + Δs >= min_group_size (required for linearization validity)."""
    m = bcm.model
    
    for group in groups:
        gk = group_key(group)
        m.addConstr(
            bcm.s[gk] + bcm.delta_s[gk] >= min_group_size,
            name=f"pos_group_{gk}"
        )


def add_coverage_constraints(
    bcm: BiasCoverageModel,
    groups: list[Group],
    label_attr: str,
    label_vals: list[str],
    grouplabel_coverage: dict[GroupLabel, int]
) -> None:
    """Add minimum coverage constraints: sy + Δsy >= m_sy."""
    for group in groups:
        gk = group_key(group)
        for label_val in label_vals:
            glk = gk + str(label_val)
            coverage_key = tuple(list(group) + [(label_attr, label_val)])
            # assign coverage requirement of 0 if empty group-label
            min_coverage = grouplabel_coverage.get(coverage_key, 0)
            
            bcm.model.addConstr(
                bcm.sy[glk] + bcm.delta_sy[glk] >= min_coverage,
                name=f"coverage_{glk}"
            )


def add_budget_constraint(
    bcm: BiasCoverageModel,
    cost_add: float,
    cost_del: float,
    budget: float
) -> None:
    """Add budget constraint: Σ (c_a·Δ⁺ + c_d·Δ⁻) <= B."""
    bcm.model.addConstr(
        gp.quicksum(
            cost_add * bcm.delta_pos[glk] + cost_del * bcm.delta_neg[glk]
            for glk in bcm.delta_sy.keys()
        ) <= budget,
        name="budget"
    )


def set_objective_min_cost(
    bcm: BiasCoverageModel,
    cost_add: float,
    cost_del: float
) -> None:
    """Minimize total cost: Σ (c_a·Δ⁺ + c_d·Δ⁻)."""
    total_cost = gp.quicksum(
        cost_add * bcm.delta_pos[glk] + cost_del * bcm.delta_neg[glk]
        for glk in bcm.delta_sy.keys()
    )
    bcm.model.setObjective(total_cost, GRB.MINIMIZE)


def set_objective_min_changes(bcm: BiasCoverageModel) -> None:
    """Minimize total changes: Σ (Δ⁺ + Δ⁻)."""
    set_objective_min_cost(bcm, cost_add=1.0, cost_del=1.0)


def set_objective_min_additions(bcm: BiasCoverageModel) -> None:
    """Minimize additions only: Σ Δ⁺."""
    set_objective_min_cost(bcm, cost_add=1.0, cost_del=0.0)


def set_objective_min_size(bcm: BiasCoverageModel) -> None:
    """Minimize total dataset size: Σ (sy + Δsy)."""
    bcm.model.setObjective(gp.quicksum(bcm.delta_sy.values()), GRB.MINIMIZE)


def extract_solution(
    bcm: BiasCoverageModel,
    grouplabel_coverage: dict[GroupLabel, int]
) -> dict[GroupLabel, int]:
    """Extract Delta_sy values from solved model."""
    result = {}
    for group_label in grouplabel_coverage.keys():
        glk = ''.join(str(attr_val[1]) for attr_val in group_label)
        # rounding applied because we solve it over the real numbers
        result[group_label] = round(bcm.delta_sy[glk].X) 
    return result


def bias_coverage_mitigation_ilp(
    df_sketch: pd.DataFrame,
    label_attr: str,
    groups: list[Group],
    grouplabel_coverage: dict[GroupLabel, int],
    eps: float = 0.1,
    objective: str = "min_changes",
    cost_add: float = 1.0,
    cost_del: float = 1.0,
    # float | None means this can be either a float or None
    # = None is the default value if the caller doesn't provide it
    budget: float | None = None,
    min_group_size: int = 1,
    verbose: bool = False
) -> dict[GroupLabel, int]:
    """
    Solve the bias-coverage optimization problem using linearized ILP.
    
    Finds adjustments Δsy such that:
      - |f_{sy}^{new} - f_y| <= eps  (fairness)
      - sy + Δsy >= m_sy             (coverage)
      - Σ(c_a·Δ⁺ + c_d·Δ⁻) <= B     (budget, if specified)
      - Objective minimized
    
    Args:
        df_sketch: DataFrame containing the data sketch
        label_attr: Name of the label attribute column
        groups: List of demographic groups as tuples of (attr, value) pairs
        grouplabel_coverage: Minimum coverage for each group-label combo
        eps: Maximum allowed deviation from demographic parity
        objective: One of "min_changes", "min_additions", "min_cost", or "min_size"
        cost_add: Cost per addition (c_a)
        cost_del: Cost per deletion (c_d)
        budget: Maximum total cost (None for unconstrained)
        min_group_size: Minimum group size after changes (for linearization validity)
        verbose: Whether to print Gurobi output
    
    Returns:
        Dictionary mapping group-label tuples to their Δ values
    """
    label_vals = list(df_sketch[label_attr].unique())

    n = get_total_count(df_sketch)
    logger.info(f"Building model with n={n}, eps={eps}")
    
    bcm = BiasCoverageModel.create()
    bcm.n = n
    
    # Build model
    add_group_label_variables(bcm, df_sketch, groups, label_vals, label_attr)
    add_label_marginals(bcm, df_sketch, label_vals, label_attr)
    add_group_variables(bcm, df_sketch, groups, label_vals)
    
    add_fairness_constraints(bcm, groups, label_vals, eps)
    add_positive_group_size_constraints(bcm, groups, min_group_size)
    add_coverage_constraints(bcm, groups, label_attr, label_vals, grouplabel_coverage)
    
    if budget is not None:
        add_budget_constraint(bcm, cost_add, cost_del, budget)
    
    # Set objective
    if objective == "min_changes":
        set_objective_min_changes(bcm)
    elif objective == "min_additions":
        set_objective_min_additions(bcm)
    elif objective == "min_cost":
        set_objective_min_cost(bcm, cost_add, cost_del)
    elif objective == "min_size":
        set_objective_min_size(bcm)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    bcm.model.setParam("OutputFlag", 1 if verbose else 0)
    bcm.model.optimize()
            
    if bcm.model.status == GRB.INFEASIBLE:
        logger.warning("Model is infeasible")
        if verbose:
            bcm.model.computeIIS()
            bcm.model.write("infeasible.ilp")
        return {}
    
    if bcm.model.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        logger.warning(f"No solution found (status={bcm.model.status})")
        return {}
    
    logger.info(f"Optimal value: {bcm.model.ObjVal:.2f}")
    return extract_solution(bcm, grouplabel_coverage)