import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg') # Set the backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import os
from onlyesm_model import SimpleEsm
import numpy as np
import logging
from focal_loss import FocalLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", total=len(dataloader))):
        if batch is None:
            logging.warning(f"Skipping empty batch at index {batch_idx}")
            continue
        fasta_embeds, annotations = batch  # Data first, labels second

        fasta_embeds = fasta_embeds.to(device)
        annotations = annotations.to(device)
        # gradient를 초기화 하는 거지 파라미터를 초기화 하진 않음
        optimizer.zero_grad()
        outputs = model(fasta_embeds)
        loss = criterion(outputs, annotations)
        loss.backward()
        optimizer.step()
        # loss는 scalar 텐서임, 텐서에서 scalar만 추출하는 게 item()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, threshold):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", total=len(dataloader))):
            if batch is None:
                 logging.warning(f"Skipping empty validation batch at index {batch_idx}")
                 continue

            fasta_embeds, annotations = batch
            fasta_embeds = fasta_embeds.to(device)
            annotations = annotations.to(device)

            outputs = model(fasta_embeds)
            loss = criterion(outputs, annotations)
            total_loss += loss.item()
            # outputs shape (Batch,outputdim)
            probabilities = torch.softmax(outputs, dim=1)
            # max_probs 해당 차원에서 가장 큰 값, preds는 가장 큰 값이 있는 인덱스
            max_probs, preds = torch.max(probabilities, dim=1)

            # Apply threshold
            preds_thresholded = preds.clone()
            preds_thresholded[max_probs < threshold] = -1  # Assign -1 to predictions below threshold
            # list에 하나씩 원소 풀어서 extend
            all_preds.extend(preds_thresholded.cpu().numpy())
            all_labels.extend(annotations.cpu().numpy())

            total_samples += annotations.size(0)
            # 예측값과 정답이 같으면 1, 다르면 0으로 카운트 그걸 sum 하면 스칼라 tensor가 나옴
            correct_predictions += (preds_thresholded == annotations).sum().item()

    all_preds_array = np.array(all_preds)
    all_labels_array = np.array(all_labels)

    # Calculate metrics
    # Use labels=np.arange(output_dim) to include all potential classes in macro averaging.
    # Use zero_division=0 to handle cases where a class has no true samples or no predictions.
    labels_for_metrics = np.arange(model.output_dim)

    precision = precision_score(
        all_labels_array,
        all_preds_array,
        labels=labels_for_metrics,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        all_labels_array,
        all_preds_array,
        labels=labels_for_metrics,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        all_labels_array,
        all_preds_array,
        labels=labels_for_metrics,
        average="macro",
        zero_division=0,
    )

    accuracy = correct_predictions / total_samples
    average_loss = total_loss / len(dataloader)

    return average_loss, accuracy, precision, recall, f1


def pad_collate_fn(batch):
    """
    Pads sequences in the batch to the maximum length in the batch.
    Filters out samples with zero-dimensional data tensors.
    Assumes that each item in the batch is a tuple (data_tensor, label_tensor).
    """
    # Filter out potential None items if any were yielded by the dataset
    batch = [item for item in batch if item is not None and item[0] is not None and item[1] is not None]
    if not batch:
        return None # Return None if the batch is empty after filtering

    try:
        data_tensors, label_tensors = zip(*batch)
    except Exception as e:
        logging.error(f"Error unpacking batch: {e}. Batch content: {batch}")
        return None # Return None if unpacking fails

    # Ensure all data tensors are at least 2D [SeqLen, EmbedDim]
    if not all(t.ndim >= 2 for t in data_tensors):
        logging.error(f"Encountered data tensor with ndim < 2 in batch.")
        return None # Or handle appropriately

    # Padding for sequence tensors (dimension 0)
    try:
        max_length = max(tensor.size(0) for tensor in data_tensors)
    except ValueError:
        logging.error(f"Could not determine max_length. Data tensors: {[t.shape for t in data_tensors]}")
        return None

    # Get embedding dimension from the first tensor (assuming consistency)
    embed_dim = data_tensors[0].size(1) if data_tensors[0].ndim >= 2 else None
    if embed_dim is None:
         logging.error("Could not determine embedding dimension from data tensors.")
         return None

    # Padding data_tensors
    padded_data = []
    for tensor in data_tensors:
        if tensor.ndim < 2 or tensor.size(1) != embed_dim:
             logging.error(f"Inconsistent tensor shape detected: {tensor.shape} vs embed_dim {embed_dim}")
             # Optionally skip this sample or return None for the batch
             continue # Skip this tensor
        padding_length = max_length - tensor.size(0)
        if padding_length > 0:
            padding = torch.zeros((padding_length, embed_dim), dtype=tensor.dtype, device=tensor.device)
            padded_tensor = torch.cat((tensor, padding), dim=0)
        else:
            padded_tensor = tensor
        padded_data.append(padded_tensor)

    if not padded_data: # If all tensors were skipped due to errors
        logging.warning("Batch resulted in no valid padded data after processing inconsistencies.")
        return None

    # Stack tensors
    try:
        batch_data = torch.stack(padded_data, dim=0)
        batch_labels = torch.stack(label_tensors, dim=0)
    except Exception as e:
        logging.error(f"Error stacking tensors: {e}. Padded data shapes: {[p.shape for p in padded_data]}, Label types: {[type(l) for l in label_tensors]}")
        return None

    return batch_data, batch_labels


class EmbedDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []  # Store labels to be used for stratification

        for filename in os.listdir(data_dir):
            if filename.endswith(".npy") and not filename.endswith("_dm.npy"):
                base_name = os.path.splitext(filename)[0]
                npy_path = os.path.join(data_dir, filename)
                header_path = os.path.join(data_dir, f"{base_name}_header.txt")

                if not os.path.exists(header_path):
                    logging.warning(f"Header file {header_path} not found for {npy_path}, skipping sample.")
                    continue

                try:
                    with open(header_path, "r") as f:
                        annotation = f.read().strip()
                    if not annotation:
                        logging.warning(f"Empty annotation in {header_path}, skipping sample.")
                        continue
                    
                    label = int(annotation)
                    self.samples.append((npy_path, header_path))
                    self.labels.append(label)

                except Exception as e:
                    logging.error(f"Error processing label for {header_path}, skipping sample: {e}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, header_path = self.samples[idx]
        label = self.labels[idx]

        # Load the numpy arrays
        data = np.load(npy_path)

        # Convert to tensors
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor

    def get_labels(self):
        return self.labels


def plot_fold_metrics(fold_idx, train_losses, val_losses, val_f1_scores, model_name, save_dir, plot_filename_base):
    """Plots and saves the training/validation metrics for a single fold."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', label=f"Fold {fold_idx+1} Training Loss")
    plt.plot(epochs, val_losses, marker='s', linestyle='--', label=f"Fold {fold_idx+1} Validation Loss")
    plt.plot(epochs, val_f1_scores, marker='^', linestyle=':', label=f"Fold {fold_idx+1} Validation F1 Score")
    plt.title(f"Training & Validation Metrics - {model_name} - Fold {fold_idx+1}")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)

    plot_save_path = os.path.join(save_dir, f"{plot_filename_base}_fold{fold_idx+1}.png")
    try:
        plt.savefig(plot_save_path)
        plt.close()
        logging.info(f"Metrics plot for fold {fold_idx+1} saved to {plot_save_path}")
    except Exception as e:
        logging.error(f"Failed to save plot for fold {fold_idx+1} to {plot_save_path}: {e}")

def plot_average_metrics(k_folds, train_losses_all, val_losses_all, val_f1_scores_all, model_name, save_dir, plot_filename_base):
    """Plots and saves the average training/validation metrics across all folds."""
    if k_folds <= 1:
        return # No average plot needed for a single fold

    # Handle potentially different lengths due to early stopping
    min_epochs = min(len(losses) for losses in train_losses_all)
    if min_epochs == 0:
        logging.warning("Cannot plot average metrics: no epochs completed in at least one fold.")
        return

    # Truncate all results to the minimum number of epochs completed across folds
    train_losses_trunc = [losses[:min_epochs] for losses in train_losses_all]
    val_losses_trunc = [losses[:min_epochs] for losses in val_losses_all]
    val_f1_scores_trunc = [scores[:min_epochs] for scores in val_f1_scores_all]

    # Calculate averages
    avg_train_losses = np.mean(train_losses_trunc, axis=0)
    avg_val_losses = np.mean(val_losses_trunc, axis=0)
    avg_val_f1_scores = np.mean(val_f1_scores_trunc, axis=0)

    epochs = range(1, min_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_losses, marker='o', linestyle='-', label="Average Training Loss")
    plt.plot(epochs, avg_val_losses, marker='s', linestyle='--', label="Average Validation Loss")
    plt.plot(epochs, avg_val_f1_scores, marker='^', linestyle=':', label="Average Validation F1 Score")
    plt.title(f"Average Training & Validation Metrics across {k_folds} Folds - {model_name}")
    plt.xlabel(f"Epochs (up to min completed: {min_epochs})")
    plt.ylabel("Average Metrics")
    plt.legend()
    plt.grid(True)

    aggregate_plot_save_path = os.path.join(save_dir, f"{plot_filename_base}_average.png")
    try:
        plt.savefig(aggregate_plot_save_path)
        plt.close()
        logging.info(f"Aggregate metrics plot saved to {aggregate_plot_save_path}")
    except Exception as e:
        logging.error(f"Failed to save aggregate plot to {aggregate_plot_save_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train your model with specific data")
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of the ESM BERT model: [8M,35M,150M,650M,3B,15B]",
    )
    parser.add_argument(
        "save_dir", type=str, help="Directory for saving model checkpoints and plots"
    )
    parser.add_argument(
        "output_dim",
        type=int,
        help="Number of groups to classify (1 for binary, >1 for multi-class)",
    )
    parser.add_argument("num_linear_layers", type=int, help="Number of linear layers to use")
    parser.add_argument("batch_size", type=int, help="Batch size to use")
    parser.add_argument("learning_rate", type=float, help="Learning rate to use")
    parser.add_argument("num_epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "threshold", type=float, help="Threshold for precision, recall, F1"
    )
    parser.add_argument(
        "optimizer",
        type=str,
        choices=["Adam", "AdamW"],
        help="Choose 'Adam' or 'AdamW' as the optimizer",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.00001, help="Weight decay value"
    )
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()

    # Print arguments for verification
    print(f"Model name: {args.model_name}")
    print(f"Save directory: {args.save_dir}")
    print(f"Output dimension: {args.output_dim}")
    print(f"Number of linear layers: {args.num_linear_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Threshold for metrics: {args.threshold}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Use GPU: {'Yes' if not args.nogpu and torch.cuda.is_available() else 'No'}")

    ########## Training Parameters #################
    output_dim = args.output_dim
    num_linear_layers = args.num_linear_layers
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    save_dir = args.save_dir
    optimizer_name = args.optimizer # Use a different name to avoid conflict with optimizer instance
    model_name = args.model_name
    threshold = args.threshold
    data_dir = f"/nashome/uglee/EnzFormer/new_embed_data/new5data/600M_4411_three"
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.nogpu else "cpu"
    )
    k_folds = 5
    os.makedirs(save_dir, exist_ok=True)

    # =========
    # Data processing
    logging.info(f"Building Dataset from {data_dir}...")
    try:
        dataset = EmbedDataset(data_dir)
        if len(dataset) == 0:
            logging.error(f"No valid samples found in {data_dir}. Exiting.")
            return # Exit if no data
    except Exception as e:
        logging.error(f"Failed to load dataset from {data_dir}: {e}")
        return
    logging.info(f"Dataset built successfully with {len(dataset)} samples.")

    ################ 5-Cross Validation #############

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_fold_best_f1 = []
    train_losses_all_folds = []
    val_losses_all_folds = []
    val_f1_scores_all_folds = []

    # Base filename for saving models and plots, incorporating hyperparameters
    run_identifier = f"{model_name}_nlayers{num_linear_layers}_bs{batch_size}_lr{learning_rate}_wd{weight_decay}_opt{optimizer_name}"
    
    dataset_indices = np.arange(len(dataset))
    dataset_labels = dataset.get_labels()

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_indices, dataset_labels)):
        logging.info(f"===== FOLD {fold+1}/{k_folds} =====")

        # Remove or adjust this line if you want to train all folds
        # if fold != 0:
        #      logging.info("Skipping fold based on conditional check.")
        #      continue

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # Use num_workers for potentially faster data loading
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
        logging.debug(f"Using {num_workers} workers for DataLoaders.")

        train_dataloader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
        )
        val_dataloader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
        )

        logging.info(f"Fold {fold+1}: Train samples={len(train_subset)}, Validation samples={len(val_subset)}")

        model = SimpleEsm(
            model_name=model_name,
            output_dim=output_dim,
            num_linear_layers=num_linear_layers,
        ).to(device)

        # Criterion
        if output_dim == 1:
            criterion = nn.BCEWithLogitsLoss().to(device)
        elif output_dim > 1:
            # Dynamically calculate class counts from the training subset for this fold
            train_labels = [dataset.labels[i] for i in train_indices]
            counts = np.bincount(train_labels, minlength=output_dim)

            # --- Consider moving class counts/weights calculation outside the fold loop if static --- 
            if len(counts) > output_dim:
                logging.warning(f"Found more classes in data ({len(counts)}) than expected ({output_dim}). Truncating.")
                counts = counts[:output_dim]
            
            # Use calculated counts for Focal Loss weights
            total_samples_in_fold = counts.sum()
            if total_samples_in_fold > 0:
                class_weights = total_samples_in_fold / (output_dim * counts + 1e-9) # Add epsilon
                alpha = torch.tensor(class_weights, dtype=torch.float32)
            else:
                logging.warning(f"Fold {fold+1} has no training samples. Using uniform weights.")
                alpha = torch.ones(output_dim, dtype=torch.float32)

            alpha = alpha.to(device)
            logging.info(f"Using Focal Loss with dynamically calculated alpha for Fold {fold+1}: {alpha.cpu().numpy()}")
            criterion = FocalLoss(gamma=2, alpha=alpha).to(device)
            # --- End of class weights section --- 
        else:
            logging.error("Invalid output_dim. Must be 1 or > 1.")
            raise ValueError("Invalid output_dim")

        # Optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            logging.error(f"Unsupported optimizer: {optimizer_name}")
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Scheduler
        warmup_steps = 5 # Consider making this an argument
        def lr_lambda(current_step):
             # Linear warmup
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1.0, warmup_steps))
            # Exponential decay after warmup (adjust decay factor 0.95 if needed)
            return 0.95 ** (current_step - warmup_steps + 1)

        scheduler = LambdaLR(optimizer, lr_lambda)

        # Early stopping setup
        early_stopping_patience = 5 # Consider making this an argument
        max_val_f1 = -np.Inf
        patience_counter = 0
        best_model_state = None
        best_epoch = -1

        fold_train_losses = []
        fold_val_losses = []
        fold_val_f1_scores = []

        logging.info(f"Starting training for fold {fold+1}...")
        for epoch in range(num_epochs):
            avg_train_loss = train(model, train_dataloader, criterion, optimizer, device)
            fold_train_losses.append(avg_train_loss)

            # Validation
            val_loss, accuracy, precision, recall, f1 = validate(
                model, val_dataloader, criterion, device, threshold
            )
            fold_val_losses.append(val_loss)
            fold_val_f1_scores.append(f1)

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
            )

            # Early Stopping Check
            if f1 > max_val_f1:
                max_val_f1 = f1
                patience_counter = 0
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
                logging.debug(f"New best F1 score ({max_val_f1:.4f}) at epoch {best_epoch}. Saving model state.")
            else:
                patience_counter += 1
                logging.debug(f"F1 did not improve. Early stopping counter: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1} due to lack of F1 improvement for {early_stopping_patience} epochs.")
                    break # Stop training this fold

            # Step the scheduler after validation
            scheduler.step()

        # --- End of Epoch Loop for Fold --- 

        all_fold_best_f1.append(max_val_f1 if best_epoch != -1 else -1)
        logging.info(f"Fold {fold+1} completed. Best Validation F1 Score = {max_val_f1:.4f} at epoch {best_epoch}")

        # Save the best model for this fold
        if best_model_state is not None:
            model_save_path = os.path.join(save_dir, f"{run_identifier}_fold{fold+1}_best_f1_epoch{best_epoch}.pth")
            try:
                torch.save(best_model_state, model_save_path)
                logging.info(f"Best model for fold {fold+1} saved to {model_save_path}")
            except Exception as e:
                logging.error(f"Failed to save best model for fold {fold+1} to {model_save_path}: {e}")
        else:
             logging.warning(f"No improvement found for fold {fold+1}, no best model saved.")

        train_losses_all_folds.append(fold_train_losses)
        val_losses_all_folds.append(fold_val_losses)
        val_f1_scores_all_folds.append(fold_val_f1_scores)

        # Plot metrics for this fold using the helper function
        plot_fold_metrics(fold, fold_train_losses, fold_val_losses, fold_val_f1_scores, model_name, save_dir, run_identifier)

        logging.info(f"---------------- End of Fold {fold+1} ----------------")


    # Aggregate results and plot average metrics using the helper function
    plot_average_metrics(k_folds, train_losses_all_folds, val_losses_all_folds, val_f1_scores_all_folds, model_name, save_dir, run_identifier)

    # Log final results
    valid_f1_scores = [f1 for f1 in all_fold_best_f1 if f1 != -1]
    if valid_f1_scores:
        avg_best_f1 = np.mean(valid_f1_scores)
        std_best_f1 = np.std(valid_f1_scores)
        logging.info(f"Average Best Validation F1 Score across {len(valid_f1_scores)} folds: {avg_best_f1:.4f} +/- {std_best_f1:.4f}")
    else:
        logging.warning("No valid best F1 scores recorded across folds.")


if __name__ == "__main__":
    main()

