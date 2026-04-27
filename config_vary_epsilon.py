"""
Configuration for vary epsilon experiments.

Define multiple runs, then select which to execute.
"""
from dataclasses import dataclass
import numpy as np
from config_datasets import SKETCHES
from bias_closed_form import get_max_bias
from pathlib import Path

@dataclass
class Config:
    # Dataset
    dataset_name: str
    sketch_path: str
    sensitive_attrs: list[str]
    label_attr: str
    
    @property
    def sketch_name(self) -> str:
        attrs = "_".join(self.sensitive_attrs)
        return f"{self.dataset_name}_{attrs}_{self.label_attr}"
    
    # Coverage
    coverage_scale: float = 0 # require X% of original counts
    min_coverage: int = 1
    plot_strip: bool = True
    
    # Experiment
    epsilons: list[float] = None
    objective: str = "min_changes"
    objectives: list[str] = None  # For comparison across objectives
    
    def __post_init__(self):
        if self.epsilons is None:
            # self.epsilons = [0.1, 0.15, 0.2, 0.3, 0.5]
            max_differential_bias = get_max_bias(self.bias_file_path, 
                                                 column="|f_sy-f_y|",
                                                 intersectional_only=True)[0]
            self.epsilons = np.linspace(0.01, max_differential_bias, 8)
        if self.objectives is None:
            self.objectives = ["min_changes", "min_size","min_additions"] # "min_additions"
    
    def output_name(self, ext: str = "csv") -> str:
        attrs = "_".join(self.sensitive_attrs)
        suffix = "_strip" if self.plot_strip else ""
        return f"{self.dataset_name}_{attrs}_{self.label_attr}_vary_epsilon_{self.objective}{suffix}.{ext}"

    def comparison_output_name(self, ext: str = "csv") -> str:
        attrs = "_".join(self.sensitive_attrs)
        suffix = "_strip" if self.plot_strip else ""
        return f"{self.dataset_name}_{attrs}_{self.label_attr}_vary_epsilon_compare_objectives{suffix}.{ext}"
    
    @property
    def bias_file_path(self)->Path:
        """Path to the bias CSV file."""
        attrs = "_".join(self.sensitive_attrs)
        return Path("results/bias_original") / f"{self.dataset_name}_{attrs}_{self.label_attr}_bias.csv"

# =============================================================================
# Define configurations
# =============================================================================

# CONFIGS = {
#     "compas_min_changes": Config(
#         dataset_name="compas",
#         sketch_path="data/compas/compas_sketch.csv",
#         sensitive_attrs=["gender", "race"],
#         label_attr="label",
#     ),
    
#     "compas_minsize": Config(
#         dataset_name="compas",
#         sketch_path="data/compas/compas_sketch.csv",
#         sensitive_attrs=["gender", "race"],
#         label_attr="label",
#         objective="min_size",
#     ),
# }

# Generate CONFIGS from SKETCHES
CONFIGS = {
    sketch.sketch_name: Config(
        dataset_name=sketch.dataset_name,
        sketch_path=str(sketch.sketch_path),
        sensitive_attrs=sketch.sensitive_attrs,
        label_attr=sketch.label_attr,
    )
    for sketch in SKETCHES
}

# CONFIGS = {}
# CONFIGS["synthetic_divergent"] = Config(
#     dataset_name="synthetic",
#     sketch_path="data/synthetic/synthetic_divergent_sketch.csv",
#     sensitive_attrs=["group"],
#     label_attr="label",
#     epsilons=[0.1, 0.15, 0.2],  
# )

# # Only certain datasets
# RUN = [sketch.sketch_name for sketch in SKETCHES if sketch.dataset_name in ["adult", "compas"]]


# =============================================================================
# Select which to run
# =============================================================================

# Option 1: Run a single config
# RUN = "compas_minsize"
# RUN = "compas_min_changes"

# Option 2: Run multiple configs
# RUN = ["compas_default", "compas_minsize", "adult_default"]

# Option 3: Run all
RUN = list(CONFIGS.keys())

# Compare objectives across epsilon sweep
COMPARE_OBJECTIVES = True