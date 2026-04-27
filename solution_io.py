# solution_io.py
import json
from pathlib import Path
from typing import Dict, Tuple, Literal
import pandas as pd

ExperimentType = Literal["vary_epsilon", "vary_budget", "vary_cost_ratio", "vary_coverage_scale"]
ObjectiveType = Literal["min_size", "min_changes", "min_additions"]

EXPERIMENT_PARAMS: dict[ExperimentType, list[str]] = {
    "vary_epsilon": ["eps"],
    "vary_budget": ["budget"],
    "vary_cost_ratio": ["cost_ratio"],
    "vary_coverage": ["coverage_scale"],
}

OBJECTIVES: list[ObjectiveType] = ["min_size", "min_changes", "min_additions"]

def serialize_solution(solution: dict) -> dict:
    """Convert solution dict to JSON-serializable format."""
    return {
        "|".join(f"{attr}={val}" for attr, val in key): change
        for key, change in solution.items()
    }


def deserialize_solution(raw: dict) -> dict:
    """Parse solution dict back to tuple keys."""
    solution = {}
    for key_str, change in raw.items():
        parts = []
        for part in key_str.split("|"):
            attr, val = part.split("=", 1)
            if val == "None":
                val = None
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass  # keep as string
            parts.append((attr, val))
        solution[tuple(parts)] = change
    return solution


def save_solution(solution: dict, path: Path) -> None:
    """Save solution to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(serialize_solution(solution), f, indent=2)


def load_solution(path: Path) -> dict:
    """Load solution from JSON file."""
    with open(path) as f:
        return deserialize_solution(json.load(f))
    

def get_additions(solution: Dict[Tuple, int]) -> Dict[Tuple, int]:
    """Extract positive changes (additions) from a solution."""
    return {gl: change for gl, change in solution.items() if change >= 0}


def get_deletions(solution: Dict[Tuple, int]) -> Dict[Tuple, int]:
    """Extract negative changes (deletions) as positive counts."""
    return {gl: -change for gl, change in solution.items() if change <= 0}


def load_mitigation_results(
    sketch, 
    experiment: ExperimentType,
    objective: ObjectiveType,
) -> pd.DataFrame:
    path = sketch.results_path(experiment)
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    df = pd.read_csv(path)
    return df[df["objective"] == objective]


def compile_solutions(
    sketch, 
    experiment: ExperimentType,
    objective: ObjectiveType,
    additions_only: bool = False,
) -> dict[tuple, dict[tuple, int]]:
    """Load all solutions keyed by experiment parameters."""
    results = load_mitigation_results(sketch, experiment, objective)
    solutions_dir = Path("results") / experiment / "solutions"
    
    param_cols = EXPERIMENT_PARAMS[experiment]
    solutions_by_params = {}
    
    for _, row in results.iterrows():
        if not row["feasible"]:
            continue
        
        params = tuple(row[col] for col in param_cols)
        solution = load_solution(solutions_dir / row["solution_file"])
        
        # Filter out groups with NaN values
        nan_groups = [gl for gl in solution if any(pd.isna(v) for _, v in gl)]
        if nan_groups:
            print(f"Warning: {len(nan_groups)} groups with NaN filtered out for {params}")
        solution = {
            gl: change for gl, change in solution.items()
            if not any(pd.isna(v) for _, v in gl)
        }
        
        if additions_only:
            solution = get_additions(solution)
        
        solutions_by_params[params] = solution
    
    return solutions_by_params