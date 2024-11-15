import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os
import optuna
from optuna.trial import TrialState
from onlyesm_model import SimpleEsm
from focal_loss import FocalLoss
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    print(f"Total number of batches in this epoch: {len(dataloader)}")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", total=len(dataloader))):
        if batch is None:
            continue
        fasta_embeds, annotations = batch

        fasta_embeds = fasta_embeds.to(device)
        annotations = annotations.to(device)

        optimizer.zero_grad()
        outputs = model(fasta_embeds)
        loss = criterion(outputs, annotations)
        loss.backward()
        # Gradient clipping (optional)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    average_loss = total_loss / len(dataloader)

    return average_loss, precision, recall, f1


def pad_collate_fn(batch):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, header_path = self.samples[idx]

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
        "save_dir", type=str, help="Directory for saving model checkpoints"
    )
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.nogpu else "cpu"
    )

    # Create the study
    study = optuna.create_study(direction="maximize")  # Changed to maximize F1 score

    # Optimize
    study.optimize(lambda trial: objective(trial, args, device), n_trials=50)

    # Print best trial
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value (Best Validation F1 Score): {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def objective(trial, args, device):
    # Suggest hyperparameters
    output_dim = 18  # Assuming this is fixed; adjust as needed
    num_blocks = trial.suggest_int("num_blocks", 2, 6)
    batch_size = 64
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-3)
    num_epochs = 50  # You can also make this a hyperparameter
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    n_head = trial.suggest_categorical("n_head", [4, 8, 12])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    early_stopping_patience = 5

    # Prepare Dataset and DataLoader
    data_dir = f"/nashome/uglee/EnzFormer/new_embed_data/{args.model_name}"
    print(f"Building Dataset from {data_dir}........................")
    dataset = EmbedDataset(data_dir)

    # For simplicity, use a single train-validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn
    )

    val_dataloader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn
    )

    # Initialize the model
    model = SimpleEsm(
        model_name=args.model_name,
        output_dim=output_dim,
    ).to(device)

    # Criterion
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
            "Invalid output_dim. It should be 2 for binary classification or greater than 2 for multi-class classification."
        )

    # Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    # Scheduler: Fixed 5% decay at each epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    best_f1_score = 0.0  # Initialize the best F1 score
    patience_counter = 0
    best_model_state = None
    threshold = 0.7
    for epoch in range(num_epochs):
        avg_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, precision, recall, f1_score = validate(model, val_dataloader, criterion, device, threshold)

        print(
            f"Trial {trial.number}, Epoch [{epoch+1}/{num_epochs}], "
            f"Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}"
        )

        # Scheduler step
        scheduler.step()

        # Early stopping based on F1 score
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered due to lack of improvement in F1 score.")
                break

        # Report to Optuna
        trial.report(f1_score, epoch)

        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Save the best model for this trial
    model_save_path = os.path.join(
        args.save_dir, f"trial_{trial.number}_best_model.pth"
    )
    torch.save(best_model_state, model_save_path)

    return best_f1_score  # Return the best F1 score for Optuna to maximize


if __name__ == "__main__":
    main()

