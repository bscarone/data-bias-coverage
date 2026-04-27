"""
Experiment: Vary cost ratio (c_d / c_a) and observe how the balance of additions vs deletions changes.
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
from config_vary_cost_ratio import CONFIGS, RUN, Config
from solution_io import save_solution
from run_utils import parse_args, should_plot

OUTPUT_DIR = Path("results/vary_cost_ratio")
SOLUTIONS_DIR = OUTPUT_DIR / "solutions"

# =============================================================================
# Data preparation
# =============================================================================

def load_and_prepare(cfg: Config) -> tuple[pd.DataFrame, list[Group], dict, dict, dict]:
    """Load data and prepare ILP inputs."""
    df = pd.read_csv(cfg.sketch_path)
    
    groups = get_groups(df, cfg.sensitive_attrs)
    
    if 'count' in df.columns:
        original_sy = df_to_grouplabel_dict(df, count_col='count')
    else:
        original_sy = get_group_counts(df, cfg.sensitive_attrs, cfg.label_attr)
    
    f_y = get_label_fractions(df, cfg.label_attr)
    
    grouplabel_coverage = {
        gl: max(cfg.min_coverage, int(cfg.coverage_scale * count))
        for gl, count in original_sy.items()
    }
    
    return df, groups, grouplabel_coverage, original_sy, f_y


# =============================================================================
# Run experiment
# =============================================================================

def run_vary_cost_ratio(
    cfg: Config,
    df: pd.DataFrame,
    groups: list[Group],
    grouplabel_coverage: dict[GroupLabel, int],
    original_sy: dict[GroupLabel, int],
    f_y: dict[str, float],
) -> pd.DataFrame:
    """Run ILP for each cost ratio value."""
    results = []
    
    for ratio in cfg.cost_ratios:
        print(f"Running cost_ratio={ratio} (c_a=1.0, c_d={ratio})...")
        
        start = time.time()
        solution = bias_coverage_mitigation_ilp(
            df_sketch=df,
            label_attr=cfg.label_attr,
            groups=groups,
            grouplabel_coverage=grouplabel_coverage,
            eps=cfg.eps,
            objective="min_cost",
            cost_add=1.0,
            cost_del=ratio,
        )
        solve_time = time.time() - start
        
        feasible = len(solution) > 0
        
        if feasible:
            # Save solution to separate file
            solution_filename = f"{cfg.sketch_name}_ratio{ratio}.json"
            save_solution(solution, SOLUTIONS_DIR / solution_filename)
            
            additions = sum(max(0, d) for d in solution.values())
            deletions = sum(max(0, -d) for d in solution.values())
            final_size = sum(original_sy[gl] + solution[gl] for gl in solution)
            total_cost = 1.0 * additions + ratio * deletions
        else:
            solution_filename = None
            additions = deletions = final_size = total_cost = 0
        
        results.append({
            'cost_ratio': ratio,
            'objective': 'min_cost',
            'cost_add': 1.0,
            'cost_del': ratio,
            'feasible': feasible,
            'solve_time': solve_time,
            'total_additions': additions,
            'total_deletions': deletions,
            'total_changes': additions + deletions,
            'total_cost': total_cost,
            'final_size': final_size,
            'solution_file': solution_filename,
        })
        
        status = "✓" if feasible else "✗ infeasible"
        print(f"  {status}, +{additions}/-{deletions}, " +
              f"cost={total_cost:.1f}, time={solve_time:.2f}s")
    
    return pd.DataFrame(results)


# =============================================================================
# Plotting
# =============================================================================

def plot_results_strip(df: pd.DataFrame, save_path: Path = None):
    """Plot mitigation strategy vs cost ratio in a compact strip layout."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.2), gridspec_kw={'wspace': 0.4})
    
    feasible = df[df['feasible']].copy()
    
    feasible['add_fraction'] = (
        feasible['total_additions'] / 
        feasible['total_changes'].replace(0, 1)
    )
    
    # Additions vs deletions
    ax = axes[0]
    ax.semilogx(feasible['cost_ratio'], feasible['total_additions'], 's-', markersize=4, label='Additions')
    ax.semilogx(feasible['cost_ratio'], feasible['total_deletions'], '^-', markersize=4, label='Deletions')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Cost ratio', fontsize=9)
    ax.set_title('Strategy shift', fontsize=10)
    ax.legend(fontsize=7, handlelength=1.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Fraction additions
    ax = axes[1]
    ax.semilogx(feasible['cost_ratio'], feasible['add_fraction'], 'o-', markersize=4)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Cost ratio', fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Addition fraction', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Total cost
    ax = axes[2]
    ax.semilogx(feasible['cost_ratio'], feasible['total_cost'], 'o-', markersize=4)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Cost ratio', fontsize=9)
    ax.set_title('Total cost', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Final size
    ax = axes[3]
    ax.semilogx(feasible['cost_ratio'], feasible['final_size'], 'o-', markersize=4)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Cost ratio', fontsize=9)
    ax.set_title('Final size', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout(pad=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.02)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_results_grid(df: pd.DataFrame, save_path: Path = None):
    """Plot mitigation strategy vs cost ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    feasible = df[df['feasible']].copy()
    
    # Compute addition fraction
    feasible['add_fraction'] = (
        feasible['total_additions'] / 
        feasible['total_changes'].replace(0, 1)
    )
    
    # Top left: Additions vs deletions
    ax = axes[0, 0]
    ax.semilogx(feasible['cost_ratio'], feasible['total_additions'], 
                's-', label='Additions')
    ax.semilogx(feasible['cost_ratio'], feasible['total_deletions'], 
                '^-', label='Deletions')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Cost ratio (c_d / c_a)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Strategy shift with cost asymmetry')
    ax.grid(True, alpha=0.3)
    
    # Top right: Fraction of additions
    ax = axes[0, 1]
    ax.semilogx(feasible['cost_ratio'], feasible['add_fraction'], 'o-')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Cost ratio (c_d / c_a)')
    ax.set_ylabel('Fraction additions')
    ax.set_ylim(0, 1)
    ax.set_title('Addition-heavy vs deletion-heavy')
    ax.grid(True, alpha=0.3)
    
    # Bottom left: Total cost
    ax = axes[1, 0]
    ax.semilogx(feasible['cost_ratio'], feasible['total_cost'], 'o-')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Cost ratio (c_d / c_a)')
    ax.set_ylabel('Total cost')
    ax.set_title('Total cost (c_a·Δ⁺ + c_d·Δ⁻)')
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Final size
    ax = axes[1, 1]
    ax.semilogx(feasible['cost_ratio'], feasible['final_size'], 'o-')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Cost ratio (c_d / c_a)')
    ax.set_ylabel('Final dataset size')
    ax.set_title('Dataset size after mitigation')
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
    print(f"Cost ratios: {cfg.cost_ratios}")
    print(f"{'='*60}")
    
    # Load data
    print(f"Loading data from {cfg.sketch_path}")
    df, groups, coverage, original_sy, f_y = load_and_prepare(cfg)
    print(f"  {sum(original_sy.values())} records, {len(groups)} groups")
    
    # Run experiment
    results = run_vary_cost_ratio(cfg, df, groups, coverage, original_sy, f_y)
    
    # Save results
    csv_path = OUTPUT_DIR / cfg.output_name("csv")
    results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    # Plot
    if should_plot():
        if cfg.plot_strip:
            plot_results_strip(results, save_path=OUTPUT_DIR / cfg.output_name("pdf"))
        else:
            plot_results_grid(results, save_path=OUTPUT_DIR / cfg.output_name("pdf"))
    
    return results


if __name__ == "__main__":
    parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle single config or list of configs
    runs = [RUN] if isinstance(RUN, str) else RUN
    
    for name in runs:
        cfg = CONFIGS[name]
        run_config(cfg)