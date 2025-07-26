import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
import logging
from tqdm import tqdm

def load_data(data_dir):
    """
    Loads data from the specified directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        tuple: A tuple containing two numpy arrays: features and labels.
    """
    features = []
    labels = []
    
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    for filename in tqdm(filenames, desc="Loading data"):
        base_name = os.path.splitext(filename)[0]
        npy_path = os.path.join(data_dir, filename)
        header_path = os.path.join(data_dir, f"{base_name}_header.txt")

        if not os.path.exists(header_path):
            logging.warning(f"Header file not found for {npy_path}, skipping.")
            continue

        try:
            # Load label
            with open(header_path, 'r') as f:
                label = int(f.read().strip())
            
            # Load features and average them
            protein_embedding = np.load(npy_path)
            # Averaging the features across the protein length
            avg_features = np.mean(protein_embedding, axis=0)
            
            features.append(avg_features)
            labels.append(label)
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")

    return np.array(features), np.array(labels)

def main():
    """
    Main function to train and evaluate the Random Forest classifier using 5-fold cross-validation.
    """
    parser = argparse.ArgumentParser(description="Train a Random Forest classifier on protein embeddings with 5-fold cross-validation.")
    parser.add_argument('--data_dir', type=str, default='/nashome/uglee/EnzFormer/new_embed_data/new5data/600M_4411_three',
                        help='Directory containing the .npy and _header.txt files.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Loading data from: {args.data_dir}")
    features, labels = load_data(args.data_dir)

    if features.shape[0] == 0:
        logging.error("No data was loaded. Please check the data directory and file formats.")
        return

    logging.info(f"Loaded {features.shape[0]} samples with feature dimension {features.shape[1]}")

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    # Lists to store metrics for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    fold_num = 1
    for train_index, test_index in tqdm(skf.split(features, labels), total=args.n_splits, desc="Cross-validation"):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        logging.info(f"--- Fold {fold_num}/{args.n_splits} ---")
        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Testing data shape: {X_test.shape}")

        # Initialize and train the Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1)
        rf_classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_classifier.predict(X_test)

        # Evaluate and store metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        
        logging.info(f"Fold {fold_num} Accuracy: {accuracies[-1]:.4f}")
        logging.info(f"Fold {fold_num} Precision: {precisions[-1]:.4f}")
        logging.info(f"Fold {fold_num} Recall: {recalls[-1]:.4f}")
        logging.info(f"Fold {fold_num} F1 Score: {f1s[-1]:.4f}")
        fold_num += 1

    # --- Overall Results ---
    print("\n--- Overall Cross-Validation Results ---")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} \u00b1 {np.std(accuracies):.4f}")
    print(f"Average Precision (Macro): {np.mean(precisions):.4f} \u00b1 {np.std(precisions):.4f}")
    print(f"Average Recall (Macro): {np.mean(recalls):.4f} \u00b1 {np.std(recalls):.4f}")
    print(f"Average F1-score (Macro): {np.mean(f1s):.4f} \u00b1 {np.std(f1s):.4f}")
    print("----------------------------------------\n")

if __name__ == '__main__':
    main()
