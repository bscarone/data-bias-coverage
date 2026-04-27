"""
Experiment: Vary coverage requirements and observe mitigation cost.
"""
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from dataframe_sketches import df_to_grouplabel_dict
from bias_coverage_ilp import bias_coverage_mitigation_ilp, Group, GroupLabel
from bias_coverage_utils import (
    get_groups, get_group_counts, get_label_fractions
)
from config_vary_coverage import CONFIGS, RUN, COMPARE_OBJECTIVES, Config
from solution_io import save_solution  
from run_utils import parse_args, should_plot, get_objectives, comparison_output_name

OUTPUT_DIR = Path("results/vary_coverage")
SOLUTIONS_DIR = OUTPUT_DIR / "solutions"  

# =============================================================================
# Data preparation
# =============================================================================

def load_data(cfg: Config) -> tuple[pd.DataFrame, list[Group], dict, dict]:
    """Load data and prepare base inputs (without coverage)."""
    df = pd.read_csv(cfg.sketch_path)
    
    groups = get_groups(df, cfg.sensitive_attrs)
    
    if 'count' in df.columns:
        original_sy = df_to_grouplabel_dict(df, count_col='count')
    else:
        original_sy = get_group_counts(df, cfg.sensitive_attrs, cfg.label_attr)
    
    f_y = get_label_fractions(df, cfg.label_attr)
    
    return df, groups, original_sy, f_y


def compute_coverage(
    original_sy: dict[GroupLabel, int],
    coverage_scale: float,
    min_coverage: int
) -> dict[GroupLabel, int]:
    """Compute coverage requirements for a given scale."""
    return {
        gl: max(min_coverage, round(coverage_scale * count))
        for gl, count in original_sy.items()
    }


# =============================================================================
# Run experiment
# =============================================================================

def run_vary_coverage(
    cfg: Config,
    df: pd.DataFrame,
    groups: list[Group],
    original_sy: dict[GroupLabel, int],
    f_y: dict[str, float],
    objective: str = None,
) -> pd.DataFrame:
    """Run ILP for each coverage scale value."""
    objective = objective or cfg.objective
    results = []
    
    for scale in cfg.coverage_scales:
        print(f"Running coverage_scale={scale}...")
        
        grouplabel_coverage = compute_coverage(original_sy, scale, cfg.min_coverage)
        
        start = time.time()
        solution = bias_coverage_mitigation_ilp(
            df_sketch=df,
            label_attr=cfg.label_attr,
            groups=groups,
            grouplabel_coverage=grouplabel_coverage,
            eps=cfg.eps,
            objective=objective,
        )
        solve_time = time.time() - start
        
        feasible = len(solution) > 0
        
        if feasible:
            # Save solution to separate file
            solution_filename = f"{cfg.sketch_name}_cov{scale}_obj{objective}.json"
            save_solution(solution, SOLUTIONS_DIR / solution_filename)
            
            additions = sum(max(0, d) for d in solution.values())
            deletions = sum(max(0, -d) for d in solution.values())
            final_size = sum(original_sy[gl] + solution[gl] for gl in solution)
        else:
            solution_filename = None
            additions = deletions = final_size = 0
        
        results.append({
            'coverage_scale': scale,
            'objective': objective,
            'feasible': feasible,
            'solve_time': solve_time,
            'total_additions': additions,
            'total_deletions': deletions,
            'total_changes': additions + deletions,
            'final_size': final_size,
            'solution_file': solution_filename,
        })
        
        status = "✓" if feasible else "✗ infeasible"
        print(f"  {status}, changes={additions + deletions}, time={solve_time:.2f}s")
    
    return pd.DataFrame(results)


def run_compare_objectives(
    cfg: Config,
    df: pd.DataFrame,
    groups: list[Group],
    original_sy: dict[GroupLabel, int],
    f_y: dict[str, float],
) -> pd.DataFrame:
    """Run coverage sweep for each objective and combine results."""
    all_results = []
    
    for objective in cfg.objectives:
        print(f"\n--- Objective: {objective} ---")
        results = run_vary_coverage(
            cfg, df, groups, original_sy, f_y,
            objective=objective
        )
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# Plotting
# =============================================================================

def plot_results(df: pd.DataFrame, save_path: Path = None):
    """Plot mitigation cost vs coverage scale."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    feasible = df[df['feasible']]
    infeasible = df[~df['feasible']]
    
    # Left: Changes vs coverage
    ax = axes[0]
    ax.plot(feasible['coverage_scale'], feasible['total_changes'],
            'o-', label='Total')
    ax.plot(feasible['coverage_scale'], feasible['total_additions'], 
            's--', label='Additions')
    ax.plot(feasible['coverage_scale'], feasible['total_deletions'], 
            '^--', label='Deletions')
    
    for _, row in infeasible.iterrows():
        ax.axvline(row['coverage_scale'], color='red', alpha=0.3, linestyle=':')
    
    ax.set_xlabel('Coverage scale')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Mitigation cost vs coverage')
    ax.grid(True, alpha=0.3)
    
    # Right: Solve time
    ax = axes[1]
    ax.semilogy(feasible['coverage_scale'], feasible['solve_time'], 'o-')
    ax.set_xlabel('Coverage scale')
    ax.set_ylabel('Solve time (s)')
    ax.set_title('Solver performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


def plot_compare_objectives_strip(df: pd.DataFrame, save_path: Path = None):
    """Plot objective comparison in a compact 1x4 strip layout."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.2), gridspec_kw={'wspace': 0.4})
    
    metrics = [
        ('total_changes', 'Total changes'),
        ('total_additions', 'Additions'),
        ('total_deletions', 'Deletions'),
        ('final_size', 'Final size'),
    ]
    
    objectives = df['objective'].unique()
    
    for ax, (metric, title) in zip(axes, metrics):
        for obj in objectives:
            obj_df = df[(df['objective'] == obj) & (df['feasible'])]
            ax.plot(obj_df['coverage_scale'], obj_df[metric], 'o-', markersize=4, label=obj)
        
        ax.set_xlabel('Coverage', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Single legend on the right
    axes[-1].legend(fontsize=7, handlelength=1.5, loc='best')
    
    plt.tight_layout(pad=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.02)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_compare_objectives_grid(df: pd.DataFrame, save_path: Path = None):
    """Plot coverage sweep comparison across objectives."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    metrics = [
        ('total_changes', 'Total changes'),
        ('total_additions', 'Additions'),
        ('total_deletions', 'Deletions'),
        ('final_size', 'Final dataset size'),
    ]
    
    objectives = df['objective'].unique()
    
    for ax, (metric, title) in zip(axes.flat, metrics):
        for obj in objectives:
            obj_df = df[(df['objective'] == obj) & (df['feasible'])]
            ax.plot(obj_df['coverage_scale'], obj_df[metric], 'o-', label=obj)
        
        ax.set_xlabel('Coverage scale')
        ax.set_ylabel(title)
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# Main
# =============================================================================

def run_config(cfg: Config):
    """Run experiment for a single configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {cfg.dataset_name} / {cfg.sensitive_attrs} / eps={cfg.eps}")
    print(f"{'='*60}")
    
    # Load data
    print(f"Loading data from {cfg.sketch_path}")
    df, groups, original_sy, f_y = load_data(cfg)
    print(f"  {sum(original_sy.values())} records, {len(groups)} groups")
    
    # Run experiment
    print(f"\nCoverage sweep: {cfg.coverage_scales}")
    results = run_vary_coverage(cfg, df, groups, original_sy, f_y)
    
    # Save results
    csv_path = OUTPUT_DIR / cfg.output_name("csv")
    results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    # Plot
    if should_plot():
        plot_results(results, save_path=OUTPUT_DIR / cfg.output_name("pdf"))
    
    return results


def run_config_compare(cfg: Config):
    """Run coverage sweep comparing multiple objectives."""
    # Override objectives from CLI if provided
    objectives = get_objectives() or cfg.objectives
    
    print(f"\n{'='*60}")
    print("Comparing objectives: " +
          f"{cfg.dataset_name} / {cfg.sensitive_attrs} / eps={cfg.eps}")
    print(f"Objectives: {objectives}")
    print(f"{'='*60}")
    
    # Load data
    print(f"Loading data from {cfg.sketch_path}")
    df, groups, original_sy, f_y = load_data(cfg)
    print(f"  {sum(original_sy.values())} records, {len(groups)} groups")
    
    # Run comparison
    print(f"\nCoverage sweep: {cfg.coverage_scales}")
    cfg.objectives = objectives
    results = run_compare_objectives(cfg, df, groups, original_sy, f_y)
    
    # Save results
    csv_name = comparison_output_name(cfg, objectives, "csv")
    csv_path = OUTPUT_DIR / csv_name
    results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    # Plot
    if should_plot():
        pdf_name = comparison_output_name(cfg, objectives, "pdf")
        if cfg.plot_strip:
            plot_compare_objectives_strip(results, save_path=OUTPUT_DIR / pdf_name)
        else:
            plot_compare_objectives_grid(results, save_path=OUTPUT_DIR / pdf_name)
    
    return results


if __name__ == "__main__":
    parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle single config or list of configs
    runs = [RUN] if isinstance(RUN, str) else RUN
    
    for name in runs:
        cfg = CONFIGS[name]
        if COMPARE_OBJECTIVES:
            run_config_compare(cfg)
        else:
            run_config(cfg)