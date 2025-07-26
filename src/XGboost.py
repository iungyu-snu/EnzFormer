import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import logging
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from imblearn.under_sampling import RandomUnderSampler

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
            avg_features = np.mean(protein_embedding, axis=0)
            
            features.append(avg_features)
            labels.append(label)
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")

    return np.array(features), np.array(labels)

def visualize_with_umap(features_np, labels_np, mode='unsupervised'):
    """
    Analyzes and visualizes npy files based on class labels using UMAP.
    
    Args:
        features_np (np.array): The feature data.
        labels_np (np.array): The label data.
        mode (str): The UMAP mode to use. One of ['unsupervised', 'supervised', 'undersample'].
    """
    try:
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        print("\nData preparation for UMAP complete:")
        print(f" - Total samples: {features_np.shape[0]}")
        print(f" - Feature vector dimension: {features_np.shape[1]}")
        label_map = {0: "Good Activity", 1: "Bad Activity", 2: "No Activity"}
        for l in sorted(label_map.keys()):
            count = counts[unique_labels == l][0] if l in unique_labels else 0
            print(f" - Class {l} ({label_map.get(l, 'Unknown')}): {count} samples")

        # 4. Run UMAP based on the selected mode
        print(f"\nStarting UMAP analysis in '{mode}' mode...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, verbose=True)
        
        title_mode = mode.capitalize()
        
        if mode == 'supervised':
            # Use labels to inform the embedding
            embedding = reducer.fit_transform(features_np, y=labels_np)
            plot_features, plot_labels = features_np, labels_np
        elif mode == 'undersample':
            # Undersample the majority class before running UMAP
            print("Performing random undersampling of the majority class...")
            rus = RandomUnderSampler(random_state=42)
            plot_features, plot_labels = rus.fit_resample(features_np, labels_np)
            print("Undersampling complete. New class distribution:")
            unique, counts = np.unique(plot_labels, return_counts=True)
            for l, c in zip(unique, counts):
                 print(f" - Class {l}: {c} samples")
            # Run unsupervised UMAP on the balanced data
            embedding = reducer.fit_transform(plot_features)
            title_mode = "Unsupervised (on Undersampled Data)"
        else: # 'unsupervised'
            embedding = reducer.fit_transform(features_np)
            plot_features, plot_labels = features_np, labels_np

        print("UMAP analysis complete.")

        # 5. Visualize the results
        print("Visualizing results and displaying plot.")
        plt.figure(figsize=(14, 10))
        
        # English labels and colors
        colors = ['green', 'orange', 'red']
        activity_labels = ['Good Activity (0)', 'Bad Activity (1)', 'No Activity (2)']
        cmap_map = ['Greens', 'Oranges', 'Reds']
        
        # Plotting order: draw the largest group (1) first, then smaller ones
        plot_order = [1, 0, 2]

        for i in plot_order:
            idx = plot_labels == i
            if not np.any(idx):
                continue
            
            points = embedding[idx]
            
            # Adjust alpha and size for the large group (class 1) to reduce clutter
            alpha = 0.3 if i == 1 else 0.7
            size = 10 if i == 1 else 15

            # Add a shaded background area using Kernel Density Estimation
            try:
                # KDE needs more than one point
                if len(points) > 1:
                    # Create a grid for KDE
                    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
                    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                                         np.linspace(y_min, y_max, 100))

                    # Fit KDE
                    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(points)
                    
                    # Evaluate density on the grid
                    Z = np.exp(kde.score_samples(np.vstack([xx.ravel(), yy.ravel()]).T))
                    Z = Z.reshape(xx.shape)

                    # Mask low-density areas to only show local background
                    # This prevents the large background color by ignoring areas below a threshold
                    Z[Z < 0.1 * Z.max()] = np.nan
                    
                    # Draw filled contours for the density
                    plt.contourf(xx, yy, Z, levels=5, cmap=cmap_map[i], alpha=0.3)
            except ValueError:
                # This can happen if all density values are under the threshold
                print(f"Warning: No significant density to plot for group {i} after thresholding.")
            except Exception as e:
                # This can fail for other reasons, e.g., singular matrix
                print(f"Warning: Could not draw density contour for group {i}. ({e})")

            # Scatter plot for the points
            plt.scatter(points[:, 0], points[:, 1], 
                        c=colors[i], 
                        label=activity_labels[i], 
                        s=size, 
                        alpha=alpha,
                        edgecolors='w', # Add white edge to points to make them pop
                        linewidths=0.3)

        # Use English for plot elements
        plt.title(f'UMAP Density Distribution ({title_mode} Mode)', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.legend(title="Activity", markerscale=2, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.show()

    except ImportError as e:
        print(f"Error: A required library is not installed: {e.name}")
        print("Please run this command in your terminal to install it:")
        print("pip install umap-learn matplotlib tqdm numpy scipy scikit-learn imbalanced-learn")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Main function to train and evaluate the XGBoost classifier using 5-fold cross-validation.
    """
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier on protein embeddings with 5-fold cross-validation.")
    parser.add_argument('--data_dir', type=str, default='/nashome/uglee/EnzFormer/new_embed_data/new5data/600M_4411_three',
                        help='Directory containing the .npy and _header.txt files.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of boosting rounds.')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Maximum tree depth.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation.')
    parser.add_argument('--visualize', type=str, default=None,
                        choices=['unsupervised', 'supervised', 'undersample'],
                        help='If specified, generates a UMAP visualization of the data using the chosen mode.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Loading data from: {args.data_dir}")
    features, labels = load_data(args.data_dir)

    if features.shape[0] == 0:
        logging.error("No data was loaded. Please check the data directory and file formats.")
        return

    logging.info(f"Loaded {features.shape[0]} samples with feature dimension {features.shape[1]}")

    if args.visualize:
        visualize_with_umap(features, labels, mode=args.visualize)
        return # Exit after visualization if that's all that's needed

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

        # Initialize and train the XGBoost Classifier
        xgb_classifier = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=args.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        xgb_classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = xgb_classifier.predict(X_test)

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