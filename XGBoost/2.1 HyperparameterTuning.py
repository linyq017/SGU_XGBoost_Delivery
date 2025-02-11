import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import optuna
import plotly
import json
import os
from datetime import datetime
"""
This script performs hyperparameter optimization for XGBoost using Optuna, a hyperparameter optimization framework.
It implements a k-fold cross-validation strategy to ensure robust model performance and uses Matthews Correlation 
Coefficient (MCC) as the optimization metric. The script handles multiclass classification problems and includes 
both tree (gbtree) and DART boosters. After finding the optimal hyperparameters, it trains a final model using 
these parameters on the full training dataset and saves it as a JSON file. The optimization process generates 
visualization plots to help understand the parameter importance and optimization history. The script also saves 
all trial results to a CSV file for further analysis.

Key features:
- Automated hyperparameter optimization using Optuna
- 5-fold cross-validation for robust performance estimation
- Interactive visualization plots of the optimization process
- Final model saving in JSON format for easy deployment
- Comprehensive logging of optimization results
- Organized output folder structure
"""


def create_output_folder(base_path):
    """Create output folder with timestamp."""
    # Create base output directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_path, f"{timestamp}")
    os.makedirs(output_path)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_path, "model"))
    os.makedirs(os.path.join(output_path, "plots"))
    os.makedirs(os.path.join(output_path, "results"))
    
    return output_path

def prepare_data(df, target_column, columns_to_drop, test_size=0.2, random_state=42):
    """Prepare data for modeling by dropping unnecessary columns."""
    # Remove the target column and other unwanted columns
    feature_columns = [c for c in df.columns if c not in columns_to_drop + [target_column]]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_columns],
        df[target_column],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )
    
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train):
    """Defines the objective function for Optuna hyperparameter optimization."""
    
    # Define the hyperparameter search space for XGBoost
    params = {
        'objective': 'multi:softmax',  # Multi-class classification with softmax output
        'eval_metric': 'mlogloss',  # Log loss for multi-class classification
        'num_class': len(np.unique(y_train)),  # Number of unique classes in y_train
        'tree_method': 'hist',  # Histogram-based optimization (efficient for large datasets)
        'lambda': trial.suggest_float("lambda", 0., 50.0),  # L2 regularization
        'alpha': trial.suggest_float("alpha", 0., 10.0),  # L1 regularization
        'booster': trial.suggest_categorical("booster", ["gbtree", "dart"]),  # Choose between tree-based and dropout-based boosting
        'max_depth': trial.suggest_int('max_depth', 4, 12),  # Max depth of trees
        'eta': trial.suggest_float('eta', 0.001, 0.3),  # Learning rate
        'gamma': trial.suggest_float('gamma', 0, 10),  # Minimum loss reduction required for a split
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),  # Fraction of features used per tree
        'subsample': trial.suggest_float('subsample', 0.4, 0.9),  # Fraction of data used per boosting round
        'min_child_weight': trial.suggest_int("min_child_weight", 0, 10)  # Minimum sum of instance weight in a child
    }
    
    # Additional hyperparameters if using 'dart' booster (dropout-based boosting)
    if params["booster"] == "dart":
        params.update({
            "sample_type": trial.suggest_categorical("sample_type", ["uniform", "weighted"]),  # Type of sampling
            "normalize_type": trial.suggest_categorical("normalize_type", ["tree", "forest"]),  # Normalization method
            "rate_drop": trial.suggest_float("rate_drop", 1e-8, 1.0, log=False),  # Dropout rate
            "skip_drop": trial.suggest_float("skip_drop", 1e-8, 1.0, log=False)  # Probability of skipping dropout
        })
    
    scores = []  # Store evaluation scores for each fold

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Ensures class distribution is maintained across folds
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        # Split training data into training and validation sets for the current fold
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Encode class labels (converting categorical labels into numerical values)
        le = LabelEncoder()
        y_fold_train = le.fit_transform(y_fold_train)
        y_fold_val = le.transform(y_fold_val)
        
        # Convert data to DMatrix format (optimized for XGBoost)
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        # Train the model using the current hyperparameters
        model = xgb.train(
            params,  # Hyperparameters
            dtrain,  # Training data
            early_stopping_rounds=20,  # Stop if no improvement after 20 rounds
            evals=[(dval, 'eval')],  # Monitor validation performance
            verbose_eval=False  # Suppress training output
        )
        
        # Predict on validation set
        y_pred = model.predict(dval)

        # Compute Matthews Correlation Coefficient (MCC) for evaluation
        scores.append(matthews_corrcoef(y_fold_val, y_pred))
    
    # Return the average MCC score across all folds (higher is better)
    return np.mean(scores)


def train_final_model(X_train, y_train, best_params):
    """Train final model with best parameters."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    dtrain = xgb.DMatrix(X_train, label=y_encoded)
    
    # Train final model
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=1000,
        early_stopping_rounds=20,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    
    return final_model

if __name__ == "__main__":
    """Main execution flow."""
    # Define input and output paths
    input_file = '/workspace/data/Akermark/MASTER_train_plotxy.csv'  # Replace with your data file
    columns_to_drop = ['REGION', 'GENERAL_TX','KARTTYP','QD_GENERAL_TX'] # Columns in the original training data that will not be used 
    target_column = 'GENERAL_TX'  # Replace with your target column
    output_folder = '/workspace/data/SGU/SFSI/SFSI/xgboostHyperparameterTuning_output'  # Base output folder name
    
    # Create output directory structure
    output_path = create_output_folder(output_folder)
    print(f"Saving results to: {output_path}")
    
    # Read your data
    print("Reading data...")
    df = pd.read_csv(input_file)
    
    # Prepare the data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, columns_to_drop,)
    
    # Create and run the optimization study
    print("Starting optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), 
                  n_trials=200, 
                  show_progress_bar=True)
    
    # Print results
    print("\nOptimization finished!")
    print(f"Best score: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = train_final_model(X_train, y_train, study.best_params)
    
    # Save model and parameters
    print("Saving model and parameters...")
    model_path = os.path.join(output_path, "model", "best_model.json")
    params_path = os.path.join(output_path, "model", "best_parameters.json")
    final_model.save_model(model_path)
    
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Save optimization results
    print("Saving optimization results...")
    results_path = os.path.join(output_path, "results", "optimization_results.csv")
    study.trials_dataframe().to_csv(results_path, index=False)
    
    # Create and save plots
    print("Creating plots...")
    plots_path = os.path.join(output_path, "plots")
    
    history_plot = optuna.visualization.plot_optimization_history(study)
    importance_plot = optuna.visualization.plot_param_importances(study)
    slice_plot = optuna.visualization.plot_slice(study)
    
    history_plot.write_html(os.path.join(plots_path, "optimization_history.html"))
    importance_plot.write_html(os.path.join(plots_path, "parameter_importance.html"))
    slice_plot.write_html(os.path.join(plots_path, "parameter_slices.html"))
    
    print("\nDone! Files have been saved in the following structure:")
    print(f"{output_path}/")
    print("├── model/")
    print("│   ├── best_model.json")
    print("│   └── best_parameters.json")
    print("├── plots/")
    print("│   ├── optimization_history.html")
    print("│   ├── parameter_importance.html")
    print("│   └── parameter_slices.html")
    print("└── results/")
    print("    └── optimization_results.csv")