from fastapi import APIRouter, HTTPException, Query
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from backend.dependencies import models, target_labels, db, model_performance_metrics, feature_importances_dict # Removed unused imports
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from pydantic import BaseModel

router = APIRouter()

# Pydantic models for response
class MetricStats(BaseModel):
    training_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1_score: float
    model_type: str
    last_trained: str
    dataset_size: int
    features_used: int
    cross_validation_folds: int
    best_params: Dict[str, Any]
    fold_accuracies: List[float]
    fold_precisions: List[float]
    fold_recalls: List[float]
    fold_f1_scores: List[float]
    confusion_matrices_per_fold: List[List[List[int]]]
    class_distribution: Dict[str, int]

class ConfusionMatrixData(BaseModel):
    confusion_matrix: List[List[int]]
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class ModelMetricsResponse(BaseModel):
    model_stats: Dict[str, MetricStats]
    confusion_matrices: Dict[str, ConfusionMatrixData]
    feature_importances: Dict[str, Dict[str, float]]

@router.get("/model-metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(model_type: str = Query("random_forest", description="Type of model to retrieve metrics for (e.g., 'random_forest', 'xgboost')")):
    print(f"DEBUG: Entering get_model_metrics function for model_type: {model_type}.")
    
    if model_type not in model_performance_metrics or not model_performance_metrics[model_type]:
        print(f"DEBUG: No performance metrics found for model_type: {model_type}. Raising 404.")
        raise HTTPException(status_code=404, detail=f"No performance metrics found for model type: {model_type}.")
    
    # Retrieve pre-calculated metrics
    metrics_raw = model_performance_metrics[model_type]
    print(f"DEBUG (model_metrics.py): metrics_raw for {model_type}: {metrics_raw}")
    
    # Manually construct MetricStats objects to ensure proper type conversion
    model_stats_to_return = {}
    for target, stats in metrics_raw.items():
        model_stats_to_return[target] = MetricStats(
            training_accuracy=stats["training_accuracy"],
            test_accuracy=stats["test_accuracy"],
            precision=stats["precision"],
            recall=stats["recall"],
            f1_score=stats["f1_score"],
            model_type=stats["model_type"],
            last_trained=stats["last_trained"],
            dataset_size=stats["dataset_size"],
            features_used=stats["features_used"],
            cross_validation_folds=stats["cross_validation_folds"],
            best_params=stats["best_params"],
            fold_accuracies=stats["fold_accuracies"],
            fold_precisions=stats["fold_precisions"],
            fold_recalls=stats["fold_recalls"],
            fold_f1_scores=stats["fold_f1_scores"],
            confusion_matrices_per_fold=stats.get("confusion_matrices_per_fold", []), # Populate new field
            class_distribution=stats.get("class_distribution", {}) # Populate new field
        )
 
    confusion_matrices_to_return = {}
    for target, stats in metrics_raw.items():
        print(f"DEBUG (model_metrics.py): Processing target {target} for confusion matrix.")
        if "confusion_matrices_per_fold" in stats and stats["confusion_matrices_per_fold"]: # Check the new field and if it's not empty
            print(f"DEBUG (model_metrics.py): Confusion matrix data found for {target}.")
            confusion_matrices_to_return[target] = ConfusionMatrixData(
                confusion_matrix=stats["confusion_matrices_per_fold"][-1], # Display the last fold's confusion matrix
                accuracy=np.mean(stats["fold_accuracies"]) if "fold_accuracies" in stats and stats["fold_accuracies"] else stats["test_accuracy"],
                precision=stats["precision"],
                recall=stats["recall"],
                f1_score=stats["f1_score"]
            )
            print(f"DEBUG (model_metrics.py): Added confusion matrix for {target}: {confusion_matrices_to_return[target].confusion_matrix}")
        else:
            print(f"DEBUG (model_metrics.py): No confusion matrix data found or it's empty for {target}.")
    
    # Combine all relevant metrics for the specified model_type
    response_data = ModelMetricsResponse(
        model_stats=model_stats_to_return,
        confusion_matrices=confusion_matrices_to_return,
        feature_importances=feature_importances_dict.get(model_type, {})
    )

    print(f"DEBUG: Returning calculated metrics for model_type: {model_type}.")
    for target, stats in model_stats_to_return.items():
        print(f"DEBUG: {target} - Test Accuracy: {stats.test_accuracy:.4f}, Precision: {stats.precision:.4f}, Recall: {stats.recall:.4f}, F1-Score: {stats.f1_score:.4f}")
    
    return response_data
 
@router.get("/feature-importances")
async def get_feature_importances(model_type: str = Query("random_forest", description="Type of model to retrieve feature importances for (e.g., 'random_forest', 'xgboost')")):
    print(f"DEBUG: Entering get_feature_importances function for model_type: {model_type}.")
    
    if model_type not in feature_importances_dict or not feature_importances_dict[model_type]:
        print(f"DEBUG: No feature importances found for model_type: {model_type}. Raising 404.")
        raise HTTPException(status_code=404, detail=f"No feature importances found for model type: {model_type}.")
    
    return convert_numpy_types(feature_importances_dict[model_type])

# The /predict-learning-style endpoint will need to be updated to select the correct model
# based on the model_type. For now, it will use the default models dictionary which should contain
# the latest trained models (which will be XGBoost).
# I will update this endpoint in a later step if explicit model selection is required for prediction.