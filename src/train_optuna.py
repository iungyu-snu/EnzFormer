import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os
import optuna
from optuna.trial import TrialState
from model import ECT

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue 

        fasta_embeds, annotations, dm_embeds = batch

        fasta_embeds = fasta_embeds.to(device)
        annotations = annotations.to(device)
        dm_embeds = dm_embeds.to(device)

        optimizer.zero_grad()
        outputs = model(fasta_embeds, dm_embeds)
        loss = criterion(outputs, annotations)
        loss.backward()
        # Gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue  

            fasta_embeds, annotations, dm_embeds = batch

            fasta_embeds = fasta_embeds.to(device)
            annotations = annotations.to(device)
            dm_embeds = dm_embeds.to(device)

            outputs = model(fasta_embeds, dm_embeds)

            # Check for NaNs in model outputs
            if torch.isnan(outputs).any():
                print(f"NaNs found in model outputs at batch {batch_idx}")
                continue

            # Compute the loss
            loss = criterion(outputs, annotations)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

def pad_collate_fn(batch):
    data_tensors, label_tensors, dm_tensors = zip(*batch)

    # Padding for sequence tensors
    max_length = max(tensor.size(0) for tensor in data_tensors + dm_tensors)

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

    # Padding dm_tensors to the same max_length
    PADDING_VALUE = -1e9
    padded_dm = []
    for tensor in dm_tensors:
        current_size = tensor.size(0)
        if current_size < max_length:
            padded_matrix = torch.full((max_length, max_length), PADDING_VALUE, dtype=torch.float32)
            padded_matrix[:current_size, :current_size] = tensor
        else:
            padded_matrix = tensor
        padded_dm.append(padded_matrix)

    # Stack tensors
    batch_data = torch.stack(padded_data, dim=0)
    batch_labels = torch.stack(label_tensors, dim=0)
    batch_dm = torch.stack(padded_dm, dim=0)
    return batch_data, batch_labels, batch_dm

class EmbedDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []

        for filename in os.listdir(data_dir):
            if filename.endswith('.npy') and not filename.endswith('_dm.npy'):
                base_name = os.path.splitext(filename)[0]
                npy_path = os.path.join(data_dir, filename)
                header_path = os.path.join(data_dir, f"{base_name}_header.txt")
                dm_path = os.path.join(data_dir, f"{base_name}_dm.npy")

                if not os.path.exists(header_path):
                    print(f"Warning: Header file {header_path} not found for {npy_path}")
                    continue

                if not os.path.exists(dm_path):
                    print(f"Warning: dm file {dm_path} not found for {npy_path}")
                    continue

                self.samples.append((npy_path, header_path, dm_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, header_path, dm_path = self.samples[idx]

        # Load the numpy arrays
        data = np.load(npy_path)
        dm_data = np.load(dm_path)

        # Load the annotation
        try:
            with open(header_path, 'r') as f:
                annotation = f.read().strip()
            if not annotation:
                raise ValueError(f"No annotation found in {header_path}")
            label = int(annotation)
        except Exception as e:
            print(f"Error processing label for sample {idx}: {e}")
            raise e

        # Convert to tensors
        data_tensor = torch.from_numpy(data).float()
        dm_tensor = torch.from_numpy(dm_data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor, dm_tensor

def main():
    parser = argparse.ArgumentParser(description="Train your model with specific data")
    parser.add_argument(
        "model_name", type=str, help="The name of the ESM BERT model: [8M,35M,150M,650M,3B,15B]"
    )
    parser.add_argument(
        "save_dir", type=str, help="Directory for saving model checkpoints"
    )
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.nogpu else "cpu")

    # Create the study
    study = optuna.create_study(direction="minimize")
    
    # Optimize
    study.optimize(lambda trial: objective(trial, args, device), n_trials=50)

    # Print best trial
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value (Best Validation Loss): {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def objective(trial, args, device):
    # Suggest hyperparameters
    output_dim = 7  # Assuming this is fixed; adjust as needed
    num_blocks = trial.suggest_int("num_blocks", 2, 6)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-3)
    num_epochs = 50  # You can also make this a hyperparameter
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    n_head = trial.suggest_categorical("n_head", [4, 8, 12])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    early_stopping_patience = 5

    # Prepare Dataset and DataLoader
    data_dir = f'/nashome/uglee/EnzFormer/embed_data/{args.model_name}'
    print(f"Building Dataset from {data_dir}........................")
    dataset = EmbedDataset(data_dir)

    # For simplicity, use a single train-validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn
    )

    # Initialize the model
    model = ECT(
        model_name=args.model_name,
        output_dim=output_dim,
        num_blocks=num_blocks,
        n_head=n_head,
        dropout_rate=dropout_rate
    ).to(device)

    # Criterion
    if output_dim == 2:
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif output_dim > 2:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError(
            "Invalid output_dim. It should be 2 for binary classification or greater than 2 for multi-class classification."
        )

    # Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler: Fixed 5% decay at each epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    min_val_loss = np.Inf
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        avg_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate(model, val_dataloader, criterion, device)

        print(
            f"Trial {trial.number}, Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        # Scheduler step
        scheduler.step()

        # Early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered due to lack of improvement.")
                break

        # Report to Optuna
        trial.report(val_loss, epoch)

        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Save the best model for this trial
    model_save_path = os.path.join(
        args.save_dir, f"trial_{trial.number}_best_model.pth"
    )
    torch.save(best_model_state, model_save_path)

    return min_val_loss

if __name__ == "__main__":
    main()

