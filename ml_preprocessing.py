"""
Dataset preprocessing module for ML pipelines.
Follows sklearn's fit/transform pattern for train/test consistency.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from typing import Optional


class DatasetPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive dataset preprocessor for ML pipelines.
    
    Parameters
    ----------
    id_columns : list[str], optional
        Column names to drop as identifiers. If None, attempts auto-detection.
    drop_high_null_cols_threshold : float, optional
        Drop columns with null fraction above this threshold before main null handling.
    null_strategy : {'drop_rows', 'drop_cols', 'impute_mean', 'impute_median', 'impute_mode'}
        Strategy for handling null values.
    null_threshold : float
        For 'drop_cols', drop columns with null fraction above this threshold.
    scaling : {'standard', 'minmax', 'robust', None}
        Scaling method for numeric features.
    encode_categoricals : bool
        Whether to one-hot encode categorical features.
    max_categories : int, optional
        Max categories per feature; rare categories become 'other'.
    feature_selection : {'variance', 'mutual_info', None}
        Feature selection method.
    variance_threshold : float
        Minimum variance for 'variance' selection.
    mi_k : int
        Number of top features to keep for 'mutual_info' selection.
    mi_task : {'classification', 'regression'}
        Task type for mutual information calculation.
    """
    
    def __init__(
        self,
        id_columns: Optional[list[str]] = None,
        drop_high_null_cols_threshold: Optional[float] = None,
        null_strategy: str = 'drop_rows',
        null_threshold: float = 0.5,
        scaling: Optional[str] = 'standard',
        encode_categoricals: bool = True,
        max_categories: Optional[int] = None,
        feature_selection: Optional[str] = None,
        variance_threshold: float = 0.0,
        mi_k: int = 10,
        mi_task: str = 'classification'
    ):
        self.id_columns = id_columns
        self.drop_high_null_cols_threshold = drop_high_null_cols_threshold
        self.null_strategy = null_strategy
        self.null_threshold = null_threshold
        self.scaling = scaling
        self.encode_categoricals = encode_categoricals
        self.max_categories = max_categories
        self.feature_selection = feature_selection
        self.variance_threshold = variance_threshold
        self.mi_k = mi_k
        self.mi_task = mi_task
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the preprocessor to the training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series, optional
            Target variable (required for mutual information selection).
        """
        X = X.copy()
        
        # Step 1: Identify columns to drop
        self._id_cols_to_drop = self._identify_id_columns(X)
        X = X.drop(columns=self._id_cols_to_drop, errors='ignore')
        
        # Step 2a: Identify high-null columns to drop
        self._high_null_cols = []
        if self.drop_high_null_cols_threshold is not None:
            null_fractions = X.isnull().mean()
            self._high_null_cols = null_fractions[null_fractions > self.drop_high_null_cols_threshold].index.tolist()
            X = X.drop(columns=self._high_null_cols)
        
        # Step 2b: Handle remaining nulls - fit imputation values
        self._fit_null_handler(X)
        X = self._transform_nulls(X)
        
        # Identify numeric and categorical columns after null handling
        self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Step 4: Fit categorical encoder
        if self.encode_categoricals and self._categorical_cols:
            self._fit_categorical_encoder(X)
            X = self._transform_categoricals(X)
        
        # Step 3: Fit scaler on numeric columns
        if self.scaling and self._numeric_cols:
            self._fit_scaler(X)
            X = self._transform_scaling(X)
        
        # Step 5: Fit feature selector
        if self.feature_selection:
            self._fit_feature_selector(X, y)
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        if not hasattr(self, '_fitted'):
            raise RuntimeError("Preprocessor must be fitted before transform.")
        
        X = X.copy()
        
        # Apply transformations in order
        X = X.drop(columns=self._id_cols_to_drop, errors='ignore')
        X = X.drop(columns=self._high_null_cols, errors='ignore')
        X = self._transform_nulls(X)
        
        if self.encode_categoricals and self._categorical_cols:
            X = self._transform_categoricals(X)
        
        if self.scaling and self._numeric_cols:
            X = self._transform_scaling(X)
        
        if self.feature_selection:
            X = self._transform_feature_selection(X)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    # --- Step 1: Identifier detection ---
    
    def _identify_id_columns(self, X: pd.DataFrame) -> list[str]:
        """Identify identifier columns to drop."""
        if self.id_columns is not None:
            return [c for c in self.id_columns if c in X.columns]
        
        # Auto-detection heuristics
        id_cols = []
        for col in X.columns:
            col_lower = col.lower()
            # Name-based detection
            if any(pattern in col_lower for pattern in ['_id', 'id_', 'identifier', 'index']):
                id_cols.append(col)
                continue
            if col_lower in ['id', 'pk', 'key', 'uuid']:
                id_cols.append(col)
                continue
            # Uniqueness-based detection (all unique values in non-small datasets)
            if len(X) > 100 and X[col].nunique() == len(X):
                if X[col].dtype == 'object' or not np.issubdtype(X[col].dtype, np.floating):
                    id_cols.append(col)
        return id_cols
    
    # --- Step 2: Null handling ---
    
    def _fit_null_handler(self, X: pd.DataFrame):
        """Fit null handling parameters."""
        if self.null_strategy == 'drop_cols':
            null_fractions = X.isnull().mean()
            self._cols_to_drop_nulls = null_fractions[null_fractions > self.null_threshold].index.tolist()
        elif self.null_strategy in ['impute_mean', 'impute_median', 'impute_mode']:
            self._impute_values = {}
            for col in X.columns:
                if X[col].isnull().any():
                    if self.null_strategy == 'impute_mean' and np.issubdtype(X[col].dtype, np.number):
                        self._impute_values[col] = X[col].mean()
                    elif self.null_strategy == 'impute_median' and np.issubdtype(X[col].dtype, np.number):
                        self._impute_values[col] = X[col].median()
                    else:  # mode for categoricals or as fallback
                        mode_result = X[col].mode()
                        self._impute_values[col] = mode_result.iloc[0] if len(mode_result) > 0 else None
    
    def _transform_nulls(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply null handling."""
        if self.null_strategy == 'drop_rows':
            return X.dropna()
        elif self.null_strategy == 'drop_cols':
            return X.drop(columns=self._cols_to_drop_nulls, errors='ignore')
        elif self.null_strategy in ['impute_mean', 'impute_median', 'impute_mode']:
            for col, value in self._impute_values.items():
                if col in X.columns and value is not None:
                    X[col] = X[col].fillna(value)
            return X
        return X
    
    # --- Step 3: Scaling ---
    
    def _fit_scaler(self, X: pd.DataFrame):
        """Fit the scaler on numeric columns."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self._scaler = scalers.get(self.scaling)
        if self._scaler and self._numeric_cols:
            numeric_data = X[self._numeric_cols]
            self._scaler.fit(numeric_data)
    
    def _transform_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to numeric columns."""
        if self._scaler and self._numeric_cols:
            cols_to_scale = [c for c in self._numeric_cols if c in X.columns]
            if cols_to_scale:
                X[cols_to_scale] = self._scaler.transform(X[cols_to_scale])
        return X
    
    # --- Step 4: Categorical encoding ---
    
    def _fit_categorical_encoder(self, X: pd.DataFrame):
        """Fit one-hot encoding parameters."""
        self._category_mappings = {}
        for col in self._categorical_cols:
            if col not in X.columns:
                continue
            categories = X[col].value_counts()
            if self.max_categories and len(categories) > self.max_categories:
                top_cats = categories.head(self.max_categories - 1).index.tolist()
                self._category_mappings[col] = top_cats + ['_other_']
            else:
                self._category_mappings[col] = categories.index.tolist()
    
    def _transform_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding."""
        dummies = []
        cols_to_drop = []
        
        for col, categories in self._category_mappings.items():
            if col not in X.columns:
                continue
            
            # Handle categories not seen during fit
            if '_other_' in categories:
                known_cats = [c for c in categories if c != '_other_']
                X[col] = X[col].apply(lambda x: x if x in known_cats else '_other_')
            
            # Create dummy columns
            dummy_df = pd.DataFrame({
                f"{col}_{cat}": (X[col] == cat).astype(int)
                for cat in categories
            }, index=X.index)
            dummies.append(dummy_df)
            cols_to_drop.append(col)
        
        X = X.drop(columns=cols_to_drop)
        if dummies:
            X = pd.concat([X] + dummies, axis=1)
        
        return X
    
    # --- Step 5: Feature selection ---
    
    def _fit_feature_selector(self, X: pd.DataFrame, y: Optional[pd.Series]):
        """Fit feature selection."""
        if self.feature_selection == 'variance':
            self._var_selector = VarianceThreshold(threshold=self.variance_threshold)
            self._var_selector.fit(X.select_dtypes(include=[np.number]))
            self._selected_features = X.select_dtypes(include=[np.number]).columns[
                self._var_selector.get_support()
            ].tolist()
            # Keep non-numeric columns
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            self._selected_features.extend(non_numeric)
            
        elif self.feature_selection == 'mutual_info':
            if y is None:
                raise ValueError("Target y required for mutual information selection.")
            
            numeric_X = X.select_dtypes(include=[np.number])
            mi_func = mutual_info_classif if self.mi_task == 'classification' else mutual_info_regression
            mi_scores = mi_func(numeric_X, y, random_state=42)
            
            mi_df = pd.DataFrame({'feature': numeric_X.columns, 'mi': mi_scores})
            mi_df = mi_df.sort_values('mi', ascending=False)
            
            k = min(self.mi_k, len(mi_df))
            self._selected_features = mi_df.head(k)['feature'].tolist()
            self._mi_scores = mi_df  # Store for inspection
    
    def _transform_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        cols_to_keep = [c for c in self._selected_features if c in X.columns]
        return X[cols_to_keep]
    
    # --- Inspection methods ---
    
    def get_dropped_id_columns(self) -> list[str]:
        """Return columns identified as identifiers."""
        return self._id_cols_to_drop if hasattr(self, '_id_cols_to_drop') else []
    
    def get_dropped_high_null_columns(self) -> list[str]:
        """Return columns dropped due to high null fraction."""
        return self._high_null_cols if hasattr(self, '_high_null_cols') else []
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return mutual information scores if computed."""
        return self._mi_scores if hasattr(self, '_mi_scores') else None