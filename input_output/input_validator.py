"""
🛡️ Input Validation Layer - Comprehensive validation for AutoML inputs
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

from .input_types import AutoMLInput, ValidationResult

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Comprehensive input validation for AutoForge AutoML
    
    Validates data quality, format, and provides corrections when possible
    """
    
    def __init__(self):
        """Initialize validator with validation rules"""
        self.validation_rules = {
            'min_samples': 10,
            'max_samples': 10000000,  # 10M samples
            'min_features': 1,
            'max_features': 10000,
            'max_missing_ratio': 0.95,
            'max_cardinality_ratio': 0.9,
            'max_memory_mb': 8192,  # 8GB
            'allowed_dtypes': ['int64', 'float64', 'object', 'category', 'bool', 'datetime64']
        }
        
        self.issues_found = []
    
    def validate(self, input_data: AutoMLInput) -> ValidationResult:
        """
        Comprehensive validation of AutoML input
        
        Args:
            input_data: AutoMLInput to validate
            
        Returns:
            ValidationResult with validation status and recommendations
        """
        logger.info("🛡️ Starting comprehensive input validation...")
        
        errors = []
        warnings = []
        recommendations = []
        corrected_data = None
        metadata = {}
        
        try:
            # Reset issues
            self.issues_found = []
            
            # Step 1: Basic structure validation
            structure_result = self._validate_structure(input_data)
            errors.extend(structure_result['errors'])
            warnings.extend(structure_result['warnings'])
            
            # Step 2: Data quality validation
            quality_result = self._validate_data_quality(input_data.data)
            errors.extend(quality_result['errors'])
            warnings.extend(quality_result['warnings'])
            recommendations.extend(quality_result['recommendations'])
            
            # Step 3: Target validation
            target_result = self._validate_target(input_data)
            errors.extend(target_result['errors'])
            warnings.extend(target_result['warnings'])
            recommendations.extend(target_result['recommendations'])
            
            # Step 4: Feature validation
            feature_result = self._validate_features(input_data)
            errors.extend(feature_result['errors'])
            warnings.extend(feature_result['warnings'])
            recommendations.extend(feature_result['recommendations'])
            
            # Step 5: Memory and performance validation
            performance_result = self._validate_performance(input_data.data)
            errors.extend(performance_result['errors'])
            warnings.extend(performance_result['warnings'])
            recommendations.extend(performance_result['recommendations'])
            
            # Step 6: Attempt corrections if possible
            if errors:
                correction_result = self._attempt_corrections(input_data, errors)
                corrected_data = correction_result['corrected_data']
                recommendations.extend(correction_result['recommendations'])
                
                # Clear errors that were corrected
                errors = [e for e in errors if e not in correction_result['corrected_errors']]
            
            # Compile metadata
            metadata = {
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'original_shape': input_data.data.shape,
                'target_column': input_data.target_column,
                'task_type': input_data.task_type,
                'data_type': input_data.data_type,
                'issues_found': len(self.issues_found),
                'corrections_applied': corrected_data is not None
            }
            
            # Determine overall validity
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info(f"✅ Input validation passed with {len(warnings)} warnings")
            else:
                logger.warning(f"⚠️  Input validation failed with {len(errors)} errors")
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                recommendations=recommendations,
                corrected_data=corrected_data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation system error: {str(e)}"],
                metadata={'validation_error': True}
            )
    
    def _validate_structure(self, input_data: AutoMLInput) -> Dict[str, List[str]]:
        """Validate basic input structure"""
        errors = []
        warnings = []
        
        # Check data type
        if not isinstance(input_data.data, pd.DataFrame):
            errors.append("Data must be a pandas DataFrame")
        
        # Check empty data
        if input_data.data.empty:
            errors.append("Dataset cannot be empty")
        
        # Check target column exists
        if input_data.target_column not in input_data.data.columns:
            errors.append(f"Target column '{input_data.target_column}' not found in data")
        
        # Check sample count
        n_samples = len(input_data.data)
        if n_samples < self.validation_rules['min_samples']:
            errors.append(f"Dataset too small: {n_samples} samples (minimum: {self.validation_rules['min_samples']})")
        
        if n_samples > self.validation_rules['max_samples']:
            warnings.append(f"Large dataset detected: {n_samples:,} samples. Consider sampling.")
        
        # Check feature count
        n_features = len(input_data.data.columns)
        if n_features < self.validation_rules['min_features']:
            errors.append(f"Too few features: {n_features} (minimum: {self.validation_rules['min_features']})")
        
        if n_features > self.validation_rules['max_features']:
            warnings.append(f"High dimensional data: {n_features} features. Consider dimensionality reduction.")
        
        # Check validation split
        if not 0 <= input_data.validation_split < 1:
            errors.append("validation_split must be between 0 and 1")
        
        if input_data.validation_split > 0.5:
            warnings.append("Large validation split. Consider using 0.2 or less.")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data quality"""
        errors = []
        warnings = []
        recommendations = []
        
        # Check missing values
        missing_ratio = data.isnull().sum().sum() / data.size
        if missing_ratio > self.validation_rules['max_missing_ratio']:
            errors.append(f"Too many missing values: {missing_ratio:.1%} (maximum: {self.validation_rules['max_missing_ratio']:.1%})")
        elif missing_ratio > 0.5:
            warnings.append(f"High missing value ratio: {missing_ratio:.1%}")
            recommendations.append("Consider robust imputation strategies")
        elif missing_ratio > 0.1:
            recommendations.append("Consider imputation for missing values")
        
        # Check for duplicate rows
        duplicate_ratio = data.duplicated().sum() / len(data)
        if duplicate_ratio > 0.5:
            warnings.append(f"Many duplicate rows: {duplicate_ratio:.1%}")
            recommendations.append("Consider removing duplicate rows")
        
        # Check data types
        invalid_dtypes = []
        for col in data.columns:
            if str(data[col].dtype) not in self.validation_rules['allowed_dtypes']:
                invalid_dtypes.append(f"{col}: {data[col].dtype}")
        
        if invalid_dtypes:
            warnings.append(f"Unusual data types detected: {', '.join(invalid_dtypes)}")
            recommendations.append("Consider data type conversion")
        
        # Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            inf_count = np.isinf(numeric_cols).sum().sum()
            if inf_count > 0:
                warnings.append(f"Infinite values detected: {inf_count} occurrences")
                recommendations.append("Replace infinite values with finite numbers")
        
        # Check for constant columns
        constant_cols = []
        for col in data.columns:
            if data[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            warnings.append(f"Constant columns detected: {', '.join(constant_cols)}")
            recommendations.append("Consider removing constant columns")
        
        return {
            'errors': errors, 
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _validate_target(self, input_data: AutoMLInput) -> Dict[str, List[str]]:
        """Validate target variable"""
        errors = []
        warnings = []
        recommendations = []
        
        target_data = input_data.get_target()
        
        # Check for missing target values
        missing_target = target_data.isnull().sum()
        if missing_target > 0:
            errors.append(f"Target column has {missing_target} missing values")
        
        # Check target cardinality
        n_unique = target_data.nunique()
        
        if input_data.task_type == 'classification':
            if n_unique < 2:
                errors.append("Classification target must have at least 2 classes")
            elif n_unique > 1000:
                warnings.append(f"High cardinality target: {n_unique} classes")
                recommendations.append("Consider regression or text classification")
            
            # Check class balance
            if n_unique <= 20:  # Reasonable number of classes to check
                class_counts = target_data.value_counts()
                min_ratio = class_counts.min() / class_counts.max()
                if min_ratio < 0.05:
                    warnings.append(f"Highly imbalanced classes: min/max ratio = {min_ratio:.3f}")
                    recommendations.append("Consider class balancing techniques")
        
        elif input_data.task_type == 'regression':
            if n_unique < 20:
                warnings.append(f"Low cardinality for regression: {n_unique} unique values")
                recommendations.append("Consider classification instead")
            
            # Check for outliers in regression target
            if target_data.dtype in ['int64', 'float64']:
                Q1 = target_data.quantile(0.25)
                Q3 = target_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((target_data < Q1 - 1.5 * IQR) | (target_data > Q3 + 1.5 * IQR)).sum()
                if outliers > len(target_data) * 0.1:
                    warnings.append(f"Many outliers in target: {outliers} ({outliers/len(target_data):.1%})")
                    recommendations.append("Consider robust regression or outlier handling")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _validate_features(self, input_data: AutoMLInput) -> Dict[str, List[str]]:
        """Validate feature columns"""
        errors = []
        warnings = []
        recommendations = []
        
        feature_data = input_data.get_features()
        
        # Check for high cardinality categorical features
        categorical_cols = feature_data.select_dtypes(include=['object', 'category'])
        for col in categorical_cols.columns:
            cardinality_ratio = feature_data[col].nunique() / len(feature_data)
            if cardinality_ratio > self.validation_rules['max_cardinality_ratio']:
                warnings.append(f"High cardinality in {col}: {cardinality_ratio:.1%}")
                recommendations.append(f"Consider encoding or grouping {col}")
        
        # Check for ID-like columns
        for col in feature_data.columns:
            col_name = str(col).lower()
            if 'id' in col_name or 'identifier' in col_name:
                unique_ratio = feature_data[col].nunique() / len(feature_data)
                if unique_ratio > 0.95:
                    warnings.append(f"Possible ID column detected: {col}")
                    recommendations.append(f"Consider excluding {col} from features")
        
        # Check for datetime columns not specified as categorical
        datetime_cols = feature_data.select_dtypes(include=['datetime64'])
        if not datetime_cols.empty and 'datetime' not in (input_data.data_type or ''):
            warnings.append(f"Datetime columns detected: {list(datetime_cols.columns)}")
            recommendations.append("Consider specifying data_type='time_series'")
        
        # Check text columns
        text_cols = []
        for col in feature_data.select_dtypes(include=['object']).columns:
            avg_length = feature_data[col].astype(str).str.len().mean()
            if avg_length > 50:
                text_cols.append(col)
        
        if text_cols and 'text' not in (input_data.data_type or ''):
            warnings.append(f"Possible text columns detected: {text_cols}")
            recommendations.append("Consider specifying data_type='text'")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _validate_performance(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate memory and performance constraints"""
        errors = []
        warnings = []
        recommendations = []
        
        # Check memory usage
        memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_mb > self.validation_rules['max_memory_mb']:
            errors.append(f"Dataset too large for memory: {memory_mb:.1f} MB (maximum: {self.validation_rules['max_memory_mb']} MB)")
        elif memory_mb > 1024:  # 1GB
            warnings.append(f"Large memory usage: {memory_mb:.1f} MB")
            recommendations.append("Consider data type optimization or sampling")
        
        # Check data sparsity
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            sparsity = (numeric_data == 0).sum().sum() / numeric_data.size
            if sparsity > 0.8:
                warnings.append(f"High data sparsity: {sparsity:.1%}")
                recommendations.append("Consider sparse data structures")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _attempt_corrections(self, input_data: AutoMLInput, 
                            errors: List[str]) -> Dict[str, Any]:
        """Attempt to correct common issues"""
        corrected_data = input_data.data.copy()
        corrected_errors = []
        recommendations = []
        
        try:
            # Correct missing target values
            if any("Target column has" in error and "missing values" in error for error in errors):
                target_mask = corrected_data[input_data.target_column].isnull()
                if target_mask.any():
                    corrected_data = corrected_data[~target_mask]
                    corrected_errors.extend([e for e in errors if "Target column has" in e and "missing values" in e])
                    recommendations.append(f"Removed {target_mask.sum()} rows with missing target values")
                    self.issues_found.append("missing_target_corrected")
            
            # Correct infinite values
            if any("Infinite values detected" in error for error in errors):
                numeric_cols = corrected_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    corrected_data[col] = corrected_data[col].replace([np.inf, -np.inf], np.nan)
                    corrected_data[col] = corrected_data[col].fillna(corrected_data[col].mean())
                
                corrected_errors.extend([e for e in errors if "Infinite values detected" in e])
                recommendations.append("Replaced infinite values with column mean")
                self.issues_found.append("infinite_values_corrected")
            
            # Correct missing values in features (simple imputation)
            if any("Too many missing values" not in error and "missing" in error.lower() for error in errors):
                numeric_cols = corrected_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if corrected_data[col].isnull().any():
                        corrected_data[col] = corrected_data[col].fillna(corrected_data[col].mean())
                
                categorical_cols = corrected_data.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    if corrected_data[col].isnull().any():
                        corrected_data[col] = corrected_data[col].fillna(corrected_data[col].mode()[0])
                
                corrected_errors.extend([e for e in errors if "missing" in e.lower() and "Too many" not in e])
                recommendations.append("Applied simple imputation for missing values")
                self.issues_found.append("missing_values_corrected")
            
            # Remove constant columns
            constant_cols = []
            for col in corrected_data.columns:
                if corrected_data[col].nunique() == 1:
                    constant_cols.append(col)
            
            if constant_cols and col != input_data.target_column:
                corrected_data = corrected_data.drop(columns=constant_cols)
                recommendations.append(f"Removed {len(constant_cols)} constant columns")
                self.issues_found.append("constant_columns_removed")
            
            # Remove duplicate rows
            duplicate_count = corrected_data.duplicated().sum()
            if duplicate_count > 0:
                corrected_data = corrected_data.drop_duplicates()
                recommendations.append(f"Removed {duplicate_count} duplicate rows")
                self.issues_found.append("duplicates_removed")
            
        except Exception as e:
            logger.warning(f"Correction attempt failed: {e}")
            recommendations.append("Automatic correction failed, manual correction required")
        
        return {
            'corrected_data': corrected_data if corrected_data is not None else input_data.data,
            'corrected_errors': corrected_errors,
            'recommendations': recommendations
        }
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get human-readable validation summary"""
        lines = []
        
        if result.is_valid:
            lines.append("✅ Input validation PASSED")
        else:
            lines.append("❌ Input validation FAILED")
        
        if result.errors:
            lines.append(f"\n🚨 Errors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Show first 5
                lines.append(f"  • {error}")
        
        if result.warnings:
            lines.append(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings[:5]:  # Show first 5
                lines.append(f"  • {warning}")
        
        if result.recommendations:
            lines.append(f"\n💡 Recommendations ({len(result.recommendations)}):")
            for rec in result.recommendations[:5]:  # Show first 5
                lines.append(f"  • {rec}")
        
        if result.corrected_data is not None:
            lines.append(f"\n🔧 Corrections applied automatically")
        
        return "\n".join(lines)
