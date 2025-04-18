# for xgboost training


import pandas as pd
import numpy as np
import xgboost as xgb
import json
import yaml
import os
import uproot
import uproot3
import awkward as ak
import argparse
from sklearn.metrics import accuracy_score, log_loss, recall_score, roc_auc_score
from sklearn.metrics import accuracy_score, log_loss, recall_score, roc_auc_score, roc_curve

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred_proba, y_pred_label, num_classes):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred_label)
    metrics["loss"] = log_loss(y_true, y_pred_proba, labels=list(range(num_classes)))
    metrics["recall"] = recall_score(y_true, y_pred_label, average="macro")
    try:
        metrics["auc"] = roc_auc_score(
            y_true, y_pred_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["auc"] = None  # Handle cases where AUC cannot be computed

    # Compute TPR and FPR for each class
    metrics["tpr"] = {}
    metrics["fpr"] = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
        metrics["tpr"][f"class_{i}"] = tpr.tolist()
        metrics["fpr"][f"class_{i}"] = fpr.tolist()

    return metrics

def save_model_and_metrics(model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgb_model.json")
    metrics_path = os.path.join(output_dir, "metrics.json")

    model.save_model(model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_predictions_to_root(df_test, y_true_bdtlabel, y_pred, output_root_path, labels, observables):
    df_out = df_test.copy()
    num_classes = len(labels)

    # One-hot ground truth
    y_true_onehot = np.eye(num_classes)[y_true_bdtlabel]
    for i, label in enumerate(labels):
        df_out[label] = y_true_onehot[:, i].astype(np.int32)

    # Predicted probabilities
    for i, label in enumerate(labels):
        df_out[f'score_{label}'] = y_pred[:, i].astype(np.float32)

    # Select columns
    columns_to_keep = labels + [f'score_{label}' for label in labels] + observables
    df_out = df_out[columns_to_keep]

    # Convert DataFrame to a dictionary of NumPy arrays
    data_dict = {col: df_out[col].values for col in df_out.columns}

    # Save to ROOT file using uproot3
    with uproot3.recreate(output_root_path) as fout:
        fout["Events"] = uproot3.newtree({col: data.dtype for col, data in data_dict.items()})
        fout["Events"].extend(data_dict)


def training_pipeline(config_path, train_csv, val_csv, test_csv, output_dir):

    print("[INFO] Starting training pipeline...")

    # Load config
    config = load_config(config_path)
    features = config["features"]
    observables = config["observables"]
    labels = config["labels"]
    target_col = config["target"]
    xgb_params = config["xgboost_params"]
    num_classes = xgb_params["num_class"]

    print("[INFO] Loaded config:", config)

    # Load data
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    print("[INFO] Loaded data:")

    X_train, y_train = df_train[features], df_train[target_col]
    X_val, y_val = df_val[features], df_val[target_col]
    X_test, y_test = df_test[features], df_test[target_col]

    print("[INFO] Data shapes:")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # Train
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=100,
        early_stopping_rounds=10,
        evals=[(dtrain, "train"), (dval, "val")]
    )

    print("[INFO] Training completed.")


    # Predict
    y_train_pred_proba = model.predict(dtrain)
    y_val_pred_proba = model.predict(dval)
    y_test_pred_proba = model.predict(dtest)

    y_train_pred_label = np.argmax(y_train_pred_proba, axis=1)
    y_val_pred_label = np.argmax(y_val_pred_proba, axis=1)
    y_test_pred_label = np.argmax(y_test_pred_proba, axis=1)

    # Compute metrics
    metrics = {
        "train": compute_metrics(y_train, y_train_pred_proba, y_train_pred_label, num_classes),
        "val": compute_metrics(y_val, y_val_pred_proba, y_val_pred_label, num_classes),
        "test": compute_metrics(y_test, y_test_pred_proba, y_test_pred_label, num_classes),
    }

    print("[INFO] Metrics:", metrics)

    # Save
    save_model_and_metrics(model, metrics, output_dir)
    output_root_path = os.path.join(output_dir, "pred.root")
    save_predictions_to_root(df_test.reset_index(drop=True), y_test.values, y_test_pred_proba, output_root_path, labels, observables)

    print("[INFO] Predictions saved to ROOT file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    args = parser.parse_args()
    # 打印现在的工作目录
    print("[INFO] Current working directory！！！！！！！！！:", os.getcwd())
    training_pipeline(args.config,
        train_csv="data/train.csv",
        val_csv="data/val.csv",
        test_csv="data/test.csv",
        output_dir="output_xgb"
    )


# class Training():
#     def __init__(self, config):
#         self.config = config
#         self.model = None
#
#     def data_preprocessing(self, df):
#         features = self.config.get('features', [])
#         target = self.config.get('target', 'bdtlabel')
#         X = df[features]
#         y = df[target]
#         return X, y
#
#     def feature_selection(self, df):
#         pass
#
#     def training(self):
#         pass
#
#     def df_to_root(self, df, output_path):
#         # convert the dataframe to ROOT file
#         pass
#
# def main(config_path):
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#
#
#
# if __name__ == "__main__":
#     pass