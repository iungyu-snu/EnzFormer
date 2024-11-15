import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from onlyesm_model import SimpleEsm
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from focal_loss import FocalLoss
import copy

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue

        fasta_embeds, annotations= batch  # Data first, labels second

        fasta_embeds = fasta_embeds.to(device)
        annotations = annotations.to(device)
        optimizer.zero_grad()
        outputs = model(fasta_embeds)
        loss = criterion(outputs, annotations)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, threshold):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            fasta_embeds, annotations = batch

            fasta_embeds = fasta_embeds.to(device)
            annotations = annotations.to(device)

            outputs = model(fasta_embeds)
            loss = criterion(outputs, annotations)
            total_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)

            # Apply threshold
            preds_thresholded = preds.clone()
            preds_thresholded[max_probs < threshold] = (
                -1
            )  # Assign -1 to predictions below threshold

            all_preds.extend(preds_thresholded.cpu().numpy())
            all_labels.extend(annotations.cpu().numpy())
            total_samples += annotations.size(0)

            correct_predictions += (preds_thresholded == annotations).sum().item()

    all_preds_array = np.array(all_preds)
    all_labels_array = np.array(all_labels)

    if len(all_preds_array) > 0:
        precision = precision_score(
            all_labels_array,
            all_preds_array,
            labels=np.unique(all_preds_array),
            average="macro",
            zero_division=0,
        )
        recall = recall_score(
            all_labels_array,
            all_preds_array,
            labels=np.unique(all_preds_array),
            average="macro",
            zero_division=0,
        )
        f1 = f1_score(
            all_labels_array,
            all_preds_array,
            labels=np.unique(all_preds_array),
            average="macro",
            zero_division=0,
        )
    else:
        precision = recall = f1 = 0.0

    accuracy = correct_predictions / total_samples
    average_loss = total_loss / len(dataloader)

    return average_loss, accuracy, precision, recall, f1


def pad_collate_fn(batch):
    """
    Pads sequences in the batch to the maximum length in the batch.
    Filters out samples with zero-dimensional data tensors.
    Assumes that each item in the batch is a tuple (data_tensor, label_tensor, dm_tensor).
    """

    data_tensors, label_tensors = zip(*batch)

    # Padding for sequence tensors
    max_length = max(tensor.size(0) for tensor in data_tensors)

    # Padding data_tensors
    padded_data = []
    for tensor in data_tensors:
        padding_length = max_length - tensor.size(0)
        if padding_length > 0:
            padding = torch.zeros((padding_length, tensor.size(1)), dtype=torch.float32)
            padded_tensor = torch.cat((tensor, padding), dim=0)
        else:
            padded_tensor = tensor
        padded_data.append(padded_tensor)


    # Stack tensors
    batch_data = torch.stack(padded_data, dim=0)
    batch_labels = torch.stack(label_tensors, dim=0)
    return batch_data, batch_labels


class EmbedDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []

        for filename in os.listdir(data_dir):
            if filename.endswith(".npy") and not filename.endswith("_dm.npy"):
                base_name = os.path.splitext(filename)[0]
                npy_path = os.path.join(data_dir, filename)
                header_path = os.path.join(data_dir, f"{base_name}_header.txt")

                if not os.path.exists(header_path):
                    print(
                        f"Warning: Header file {header_path} not found for {npy_path}"
                    )
                    continue
                self.samples.append((npy_path, header_path))
                #self.samples.append((npy_path, header_path, dm_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, header_path = self.samples[idx]
       # npy_path, header_path, dm_path = self.samples[idx]

        # Load the numpy arrays
        data = np.load(npy_path)

        # Load the annotation
        try:
            with open(header_path, "r") as f:
                annotation = f.read().strip()
            if not annotation:
                raise ValueError(f"No annotation found in {header_path}")
            label = int(annotation)
        except Exception as e:
            print(f"Error processing label for sample {idx}: {e}")
            raise e

        # Convert to tensors
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor

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
    parser.add_argument("num_blocks", type=int, help="Number of linear blocks to use")
    parser.add_argument("batch_size", type=int, help="Batch size to use")
    parser.add_argument("learning_rate", type=float, help="Learning rate to use")
    parser.add_argument("num_epochs", type=int, help="Number of epochs")
    parser.add_argument("n_head", type=int, help="Number of heads for attention")
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
        "--dropout_rate", type=float, default=0.1, help="Dropout rate to use"
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
    print(f"Number of blocks: {args.num_blocks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Number of heads for Attention: {args.n_head}")
    print(f"Threshold for metrics: {args.threshold}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Use GPU: {'Yes' if not args.nogpu and torch.cuda.is_available() else 'No'}")

    ########## Training Parameters #################
    output_dim = args.output_dim
    num_blocks = args.num_blocks
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    dropout_rate = args.dropout_rate
    weight_decay = args.weight_decay
    save_dir = args.save_dir
    model_name = args.model_name
    n_head = args.n_head
    threshold = args.threshold
    data_dir = f"/nashome/uglee/EnzFormer/embed_data/{model_name}"
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.nogpu else "cpu"
    )
    k_folds = 5
    os.makedirs(save_dir, exist_ok=True)
    
    

    # ========
    #Data processing
    print(f"Building Dataset from {data_dir}........................")
    dataset = EmbedDataset(data_dir)
    if len(dataset) == 0:
        raise ValueError(f"No valid samples found in {data_dir}.")

    ################ 5-Cross Validation #############

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    train_losses_all_folds = []
    val_losses_all_folds = []

    print("Training starts")
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        if fold != 0:
            continue
        
        
        
        
        print(f"FOLD {fold+1}/{k_folds}")
        print("--------------------------------")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_dataloader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn
        )

        val_dataloader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn
        )

        model = SimpleEsm(
            model_name=model_name,
            output_dim=output_dim,
        ).to(device)

        # Define criterion based on output_dim
        if output_dim == 1:
            criterion = nn.BCEWithLogitsLoss().to(device)
        elif output_dim > 1:
            counts = [
            11227,  # Class 0
            6278,   # Class 1
            25214,  # Class 2
            4628,   # Class 3
            2768,   # Class 4
            2692,   # Class 5
            5542,   # Class 6
            411,    # Class 7
            27,     # Class 8
            954,    # Class 9
            9261,   # Class 10
            2293,   # Class 11
            279,    # Class 12
            261,    # Class 13
            175,    # Class 14
            116,    # Class 15
            17207,  # Class 16
            475     # Class 17
            ]
            counts = np.array(counts, dtype=np.float32)
            num_classes = len(counts)
            total_samples = counts.sum()
        
            class_weights = total_samples / (num_classes * counts)
            max_weight = class_weights.max()
            normalized_weights = class_weights / max_weight
            min_weight = 0.1
            normalized_weights = np.clip(normalized_weights, min_weight, None)
            alpha = torch.tensor(normalized_weights, dtype=torch.float32)
            alpha = alpha.to(device)
            criterion = FocalLoss(gamma=2, alpha=alpha).to(device)
            
        else:
            raise ValueError(
                "Invalid output_dim. It should be 1 for binary classification or greater than 1 for multi-class classification."
            )

        # Define optimizer
        if args.optimizer == "Adam":
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif args.optimizer == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        # Define scheduler with warm-up and decay
        warmup_steps = 5

        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return float(epoch + 1) / float(max(1, warmup_steps))
            return 0.95 ** (epoch - warmup_steps + 1)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # Initialize early stopping variables for this fold
        early_stopping_patience = 5
        best_f1 = -float('inf')
        patience_counter = 0
        best_model_state = None

        fold_train_losses = []
        fold_val_losses = []

        for epoch in range(num_epochs):
            avg_loss = train(model, train_dataloader, criterion, optimizer, device)
            fold_train_losses.append(avg_loss)
            val_loss, accuracy, precision, recall, f1 = validate(
                model, val_dataloader, criterion, device, threshold
            )
            fold_val_losses.append(val_loss)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
            )

            # Early Stopping Check
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                best_model_state = model.state_dict()
                print(
                    f"New best F1 score: {best_f1:.4f}. "
                    f"Saving model and resetting patience counter."
                )
            else:
                patience_counter += 1
                print(
                    f"Early stopping counter: {patience_counter} / {early_stopping_patience}"
                )
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered due to lack of improvement.")
                    break

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # Save the best model for this fold
        if best_model_state is not None:
            model_save_path = os.path.join(
                save_dir,
                f"{model_name}_fold{fold+1}_blocks{num_blocks}_lr{learning_rate}_dropout{dropout_rate}_wd{weight_decay}_earlystopped.pth",
            )
            torch.save(best_model_state, model_save_path)
            print(f"Best model saved to {model_save_path}")
        else:
            model_save_path = os.path.join(
                save_dir,
                f"{model_name}_fold{fold+1}_blocks{num_blocks}_lr{learning_rate}_dropout{dropout_rate}_wd{weight_decay}.pth",
            )
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        train_losses_all_folds.append(fold_train_losses)
        val_losses_all_folds.append(fold_val_losses)

        # Plotting for this fold
        plt.figure()
        plt.plot(fold_train_losses, label=f"Fold {fold+1} Training Loss")
        plt.plot(fold_val_losses, label=f"Fold {fold+1} Validation Loss")

        plt.title(f"Training and Validation Loss for {model_name}_Fold{fold+1}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plot_save_path = os.path.join(
            save_dir,
            f"{model_name}_fold{fold+1}_blocks{num_blocks}_lr{learning_rate}_dropout{dropout_rate}_wd{weight_decay}.png",
        )
        plt.savefig(plot_save_path)
        plt.close()
        print(f"Loss plot saved to {plot_save_path}")

    #    print(
    #        f"Cross-validation complete for fold {fold+1}. Validation Loss = {min_val_loss:.4f}"
    #    )
        print("--------------------------------------------------------\n")

    # Optionally, aggregate results across folds
    if k_folds > 1:
        avg_train_losses = np.mean(train_losses_all_folds, axis=0)
        avg_val_losses = np.mean(val_losses_all_folds, axis=0)

        plt.figure()
        plt.plot(avg_train_losses, label="Average Training Loss")
        plt.plot(avg_val_losses, label="Average Validation Loss")

        plt.title(
            f"Average Training and Validation Loss across {k_folds} Folds for {model_name}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        aggregate_plot_save_path = os.path.join(
            save_dir,
            f"{model_name}_blocks{num_blocks}_lr{learning_rate}_dropout{dropout_rate}_wd{weight_decay}_average.png",
        )
        plt.savefig(aggregate_plot_save_path)
        plt.close()
        print(f"Aggregate loss plot saved to {aggregate_plot_save_path}")


if __name__ == "__main__":
    main()
