"""
Model Training Script
Trains a Random Forest classifier on normalized hand landmark coordinates,
evaluates cross-validation performance, and exports model weights for production inference.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train():
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "landmarks_dataset.pickle")
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    model_path = os.path.join(model_dir, "gesture_model.p")

    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset not found at {dataset_path}. Please run collect_data.py or scripts/generate_canonical_dataset.py first.")
        return

    print(f"[Training] Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    data = np.asarray(dataset['data'])
    labels = np.asarray(dataset['labels'])

    print(f"[Training] Total samples: {len(data)}, Feature dimension: {data.shape[1]}")
    unique_classes, counts = np.unique(labels, return_counts=True)
    print(f"[Training] Classes ({len(unique_classes)}): {list(unique_classes)}")

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    print(f"[Training] Training Random Forest model on {len(x_train)} samples...")
    model = RandomForestClassifier(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print(f" MODEL EVALUATION REPORT (Test Accuracy: {acc * 100:.2f}%)")
    print("="*50)
    print(classification_report(y_test, y_pred))

    # Save model and classes
    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'classes': model.classes_,
            'accuracy': acc
        }, f)

    print(f"[Training] Trained model successfully exported to {model_path}!")


if __name__ == "__main__":
    train()
