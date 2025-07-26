import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import argparse
from imblearn.under_sampling import RandomUnderSampler

def analyze_npy_distribution(mode='unsupervised'):
    """
    Analyzes and visualizes npy files based on class labels using UMAP.
    
    Args:
        mode (str): The UMAP mode to use. One of ['unsupervised', 'supervised', 'undersample'].
    """
    try:
        # 1. Find all *_header.txt files
        header_files = glob.glob('*_header.txt')
        if not header_files:
            print("Could not find '_header.txt' files in the current directory.")
            print("Please check if the file naming format is correct. (e.g., mydata_header.txt)")
            return
            
        print(f"Found {len(header_files)} header files.")

        features = []
        labels = []
        
        # 2. Read files and prepare data
        for header_file in tqdm(header_files, desc="Processing files"):
            basename = header_file.replace('_header.txt', '')
            npy_file = f"{basename}.npy"
            
            if os.path.exists(npy_file):
                try:
                    # Read label
                    with open(header_file, 'r') as f:
                        label = int(f.read().strip())
                    
                    # Read npy file and process (using mean)
                    data = np.load(npy_file)
                    feature_vector = np.mean(data, axis=0)
                    
                    features.append(feature_vector)
                    labels.append(label)
                except Exception as e:
                    print(f"Error occurred while processing file ({header_file}): {e}")
            else:
                print(f"Warning: Could not find {npy_file} file.")

        if not features:
            print("Could not find any data to process. Please check if npy files exist.")
            return

        # 3. Prepare data for UMAP
        features_np = np.array(features)
        labels_np = np.array(labels)
        
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        print("\nData preparation complete:")
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run UMAP analysis on protein embedding data.")
    parser.add_argument('--mode', type=str, default='unsupervised',
                        choices=['unsupervised', 'supervised', 'undersample'],
                        help='The mode for UMAP analysis.')
    args = parser.parse_args()
    
    analyze_npy_distribution(mode=args.mode)
