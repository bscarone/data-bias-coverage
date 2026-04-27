"""
Dataset configurations for sketch and Uniform Bias computation.

Defines DatasetConfig, which includes the sensitive attributes and labels and the collection of datasets to analyze.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ExperimentType = Literal["vary_epsilon", "vary_budget", "vary_cost_ratio", "vary_coverage_scale"]


@dataclass
class DatasetConfig:
    dataset_name: str
    # sensitive_attrs: list[str]
    # label_attr: str
    # add other attributes as needed

    @property
    def data_dir(self) -> Path:
        """Directory containing all dataset artifacts."""
        return Path("data") / self.dataset_name

    @property
    def raw_path(self) -> Path:
        """Path to the raw data CSV file."""
        return self.data_dir / f"{self.dataset_name}_raw.csv"


@dataclass
class SketchConfig:
    dataset_name : str
    sensitive_attrs: list[str]
    label_attr: str
    include_missing: bool = False
    # add other attributes as needed
            
    @property
    def data_dir(self) -> Path:
        """Directory containing all dataset artifacts."""
        return Path("data") / self.dataset_name
            
    @property
    def data_path(self) -> Path:
        """Path to the raw data CSV file."""
        return self.data_dir / f"{self.dataset_name}_raw.csv"
    
    @property
    def drop_nulls_path(self) -> Path:
        """Path to data with nulls removed (for sampling)."""
        return self.data_dir / f"{self.dataset_name}_drop_nulls_processed.csv"

    @property
    def ml_ready_path(self) -> Path:
        """Path to fully preprocessed data (encoded, scaled, etc. for ML)."""
        return self.data_dir / f"{self.dataset_name}_ml_ready.csv"
            
    @property
    def sketch_name(self) -> str:
        attrs = "_".join(self.sensitive_attrs)
        return f"{self.dataset_name}_{attrs}_{self.label_attr}"
    
    @property
    def columns(self) -> list[str]:
        return self.sensitive_attrs + [self.label_attr]

    @property
    def sketch_path(self) -> Path:
        """Path to the sketch CSV file."""
        return self.data_dir / f"{self.sketch_name}_sketch.csv"
    
    def results_path(self, experiment: ExperimentType) -> Path:
        """Path to results for a given experiment type."""
        return Path("results") / experiment / f"{self.sketch_name}_{experiment}_compare_objectives.csv"
    
    @property
    def bias_file_path(self)->Path:
        """Path to the bias CSV file."""
        return Path("results/bias_original") / f"{self.sketch_name}_bias.csv"

    
DATASETS = [
    DatasetConfig(
        dataset_name="compas"
    ),
    DatasetConfig(
        dataset_name="adult"
    ),
]
    
SKETCHES = [
    SketchConfig(
        dataset_name="compas",
        sensitive_attrs=["Sex_Code_Text","race_binary"],
        label_attr="ScoreText",
    ),
    SketchConfig(
        dataset_name="adult",
        sensitive_attrs=["sex","race_binary"],
        label_attr="income",
    ),
    SketchConfig(
        dataset_name="default_credit",
        sensitive_attrs=["sex_label","EDUCATION"],
        label_attr="default payment next month",
    ),
]
