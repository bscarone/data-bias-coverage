"""
Configuration management for dataset preprocessing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class PreprocessingConfig:
    """Documents preprocessing applied to a dataset."""
    
    # Dataset identification
    dataset_name: str
    raw_data_path: str
    target_column: Optional[str] = None
    version: str = "1.0"
    
    # Step 1: ID columns
    id_columns: Optional[list[str]] = None
    auto_detect_ids: bool = True
    
    # Step 2: Null handling
    drop_high_null_cols_threshold: Optional[float] = None
    null_strategy: str = "drop_rows"
    null_threshold: float = 0.5
    
    # Step 3: Scaling
    scaling: Optional[str] = "standard"
    
    # Step 4: Encoding
    encode_categoricals: bool = True
    max_categories: Optional[int] = None
    
    # Step 5: Feature selection
    feature_selection: Optional[str] = None
    variance_threshold: float = 0.0
    mi_k: int = 10
    mi_task: str = "classification"
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: Optional[str] = None
    
    @classmethod
    def drop_nulls_only(
        cls,
        dataset_name: str,
        raw_data_path: str,
        target_column: Optional[str] = None,
        drop_high_null_cols_threshold: Optional[float] = None,
        null_strategy: str = "drop_rows",
        null_threshold: float = 0.5
    ) -> "PreprocessingConfig":
        """Create a config that only drops nulls (for pre-sampling cleanup)."""
        return cls(
            dataset_name=dataset_name,
            raw_data_path=raw_data_path,
            target_column=target_column,
            id_columns=[],
            auto_detect_ids=False,
            drop_high_null_cols_threshold=drop_high_null_cols_threshold,
            null_strategy=null_strategy,
            null_threshold=null_threshold,
            scaling=None,
            encode_categoricals=False,
            feature_selection=None
        )
    
    def to_preprocessor(self):
        """Create a DatasetPreprocessor from this config."""
        from ml_preprocessing import DatasetPreprocessor
        
        return DatasetPreprocessor(
            id_columns=self.id_columns,
            drop_high_null_cols_threshold=self.drop_high_null_cols_threshold,
            null_strategy=self.null_strategy,
            null_threshold=self.null_threshold,
            scaling=self.scaling,
            encode_categoricals=self.encode_categoricals,
            max_categories=self.max_categories,
            feature_selection=self.feature_selection,
            variance_threshold=self.variance_threshold,
            mi_k=self.mi_k,
            mi_task=self.mi_task
        )


@dataclass
class PreprocessingResult:
    """Records what actually happened during preprocessing."""
    
    config: PreprocessingConfig
    
    # What was learned/applied
    dropped_id_columns: list[str] = field(default_factory=list)
    dropped_null_columns: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    numeric_columns_scaled: list[str] = field(default_factory=list)
    categorical_columns_encoded: list[str] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Preprocessing: {self.config.dataset_name} v{self.config.version}",
            f"Rows: {self.rows_before} → {self.rows_after} ({self.rows_after - self.rows_before:+d})",
            f"Columns: {self.columns_before} → {self.columns_after}",
        ]
        if self.dropped_id_columns:
            lines.append(f"Dropped IDs: {self.dropped_id_columns}")
        if self.selected_features:
            lines.append(f"Selected {len(self.selected_features)} features")
        return "\n".join(lines)
    
    
PREPROCESSING_CONFIGS = {
    "compas": PreprocessingConfig(
        dataset_name="compas",
        raw_data_path="data/compas/compas_raw.csv",
        target_column="ScoreText",
        id_columns=[
            # IDs
            "Person_ID", "AssessmentID", "Case_ID", "ScaleSet_ID", "Scale_ID",
            # Date/name columns
            "LastName", "FirstName", "DateOfBirth", "Screening_Date",
            # Leaky columns (derived from target)
            "RawScore", "DecileScore", "RecSupervisionLevel", "RecSupervisionLevelText",
            # Probably not useful
            "IsCompleted", "IsDeleted",
        ],
    ),
}

# config = PREPROCESSING_CONFIGS[dataset_name]
# preprocessor = config.to_preprocessor()