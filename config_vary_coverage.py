"""
Configuration for vary coverage experiments.

Define multiple runs, then select which to execute.
"""
from dataclasses import dataclass
from config_datasets import SKETCHES

@dataclass
class Config:
    # Dataset
    dataset_name: str
    sketch_path: str
    sensitive_attrs: list[str]
    label_attr: str
    
    # Fixed fairness tolerance
    eps: float = 0.05
    
    # Coverage sweep
    coverage_scales: list[float] = None
    min_coverage: int = 1
    
    # Experiment
    objective: str = "min_changes"
    objectives: list[str] = None  # For comparison across objectives
    plot_strip: bool = True
    
    def __post_init__(self):
        if self.coverage_scales is None:
            self.coverage_scales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                    0.6, 0.7, 0.8, 0.9, 1.0]
        if self.objectives is None:
            self.objectives = ["min_changes", "min_size"] # "min_additions"
    
    def output_name(self, ext: str = "csv") -> str:
        attrs = "_".join(self.sensitive_attrs)
        eps_str = str(self.eps).replace(".", "")
        suffix = "_strip" if self.plot_strip else ""
        return f"{self.dataset_name}_{attrs}"\
            f"_vary_coverage_eps{eps_str}_{self.objective}{suffix}.{ext}"

    def comparison_output_name(self, ext: str = "csv") -> str:
        attrs = "_".join(self.sensitive_attrs)
        suffix = "_strip" if self.plot_strip else ""
        return f"{self.dataset_name}_{attrs}_{self.label_attr}_vary_coverage"\
            +f"_compare_objectives{suffix}.{ext}"
            
    @property
    def sketch_name(self) -> str:
        return f"{self.dataset_name}_{'_'.join(self.sensitive_attrs)}_{self.label_attr}"


# =============================================================================
# Define configurations
# =============================================================================

# CONFIGS = {
#     "compas_default": Config(
#         dataset_name="compas",
#         sketch_path="data/compas/compas_sketch.csv",
#         sensitive_attrs=["gender", "race"],
#         label_attr="label",
#         eps=0.05,
#     ),
    
#     "compas_strict": Config(
#         dataset_name="compas",
#         sketch_path="data/compas/compas_sketch.csv",
#         sensitive_attrs=["gender", "race"],
#         label_attr="label",
#         eps=0.02,
#     ),
    
#     "compas_loose": Config(
#         dataset_name="compas",
#         sketch_path="data/compas/compas_sketch.csv",
#         sensitive_attrs=["gender", "race"],
#         label_attr="label",
#         eps=0.1,
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
# )

# =============================================================================
# Select which to run
# =============================================================================

# Option 1: Run a single config
# RUN = "compas_default"

# Option 2: Run multiple configs
# RUN = ["compas_default", "compas_strict", "adult_default"]

# Option 3: Run all
RUN = list(CONFIGS.keys())

# Compare objectives across coverage sweep
COMPARE_OBJECTIVES = True