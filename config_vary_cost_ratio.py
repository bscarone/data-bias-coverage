"""
Configuration for vary cost ratio experiments.

Varies c_d/c_a ratio to see how cost asymmetry affects the balance
of additions vs deletions.

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
    plot_strip: bool = True
    
    # Fixed parameters
    eps: float = 0.05
    coverage_scale: float = 0.0
    min_coverage: int = 1
    
    # Cost ratio sweep (c_d / c_a, with c_a fixed at 1.0)
    cost_ratios: list[float] = None
    
    def __post_init__(self):
        if self.cost_ratios is None:
            self.cost_ratios = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    
    def output_name(self, ext: str = "csv") -> str:
        attrs = "_".join(self.sensitive_attrs)
        suffix = "_strip" if self.plot_strip else ""
        return f"{self.dataset_name}_{attrs}_{self.label_attr}"\
            +f"_vary_cost_ratio_compare_objectives{suffix}.{ext}"
            
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
    
#     "compas_fine_ratios": Config(
#         dataset_name="compas",
#         sketch_path="data/compas/compas_sketch.csv",
#         sensitive_attrs=["gender", "race"],
#         label_attr="label",
#         eps=0.05,
#         cost_ratios=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
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