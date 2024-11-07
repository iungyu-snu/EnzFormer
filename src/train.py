import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from model import ECT
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score



def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue 

        fasta_embeds, annotations, dm_embeds = batch  # Data first, labels second

        fasta_embeds = fasta_embeds.to(device)
        annotations = annotations.to(device)
        dm_embeds = dm_embeds.to(device)
        optimizer.zero_grad()
        outputs = model(fasta_embeds, dm_embeds)
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

            fasta_embeds, annotations, dm_embeds = batch

            fasta_embeds = fasta_embeds.to(device)
            annotations = annotations.to(device)
            dm_embeds = dm_embeds.to(device)

            outputs = model(fasta_embeds, dm_embeds)

            if torch.isnan(outputs).any():
                print(f"NaNs found in model outputs at batch {batch_idx}")
                continue


            loss = criterion(outputs, annotations)
            total_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)

            # Apply threshold
            preds_thresholded = preds.clone()
            preds_thresholded[max_probs < threshold] = -1  # Assign -1 to predictions below threshold

            all_preds.extend(preds_thresholded.cpu().numpy())
            all_labels.extend(annotations.cpu().numpy())
            total_samples += annotations.size(0)

            # For computing accuracy
            correct_predictions += (preds_thresholded == annotations).sum().item()

    all_preds_array = np.array(all_preds)
    all_labels_array = np.array(all_labels)

    # valid_indicies => boolean for valid data
#    valid_indices = all_preds_array != -1
#    valid_preds = all_preds_array[valid_indices]
#    valid_labels = all_labels_array[valid_indices]

    if len(all_preds_array) > 0:
        precision = precision_score(all_labels_array, all_preds_array, labels=np.unique(all_preds_array), average='macro', zero_division=0)
        recall = recall_score(all_labels_array, all_preds_array, labels=np.unique(all_preds_array), average='macro', zero_division=0)
        f1 = f1_score(all_labels_array, all_preds_array, labels=np.unique(all_preds_array), average='macro', zero_division=0) 
    else:
        precision = recall = f1 = 0.0


    accuracy = correct_predictions / total_samples
    average_loss = total_loss / len(dataloader)

    return average_loss, accuracy, precision, recall, f1














def pad_collate_fn(batch):
    """
    Pads sequences in the batch to the maximum length in the batch.
    Filters out samples with zero-dimensional data tensors.
    Assumes that each item in the batch is a tuple (data_tensor, label_tensor).
    """

    data_tensors, label_tensors, dm_tensors = zip(*batch)

    # ====
    # padding for seq tensors
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
    parser.add_argument("output_dim", type=int, help="Number of groups to classify")
    parser.add_argument("num_blocks", type=int, help="Number of linear blocks to use")
    parser.add_argument("batch_size", type=int, help="Batch size to use")
    parser.add_argument("learning_rate", type=float, help="Learning rate to use")
    parser.add_argument("num_epochs", type=int, help="Number of epochs")
    parser.add_argument("n_head", type=int, help="head_num for attention")
    parser.add_argument("threshold", type=float, help="threshold ofor precision,recall,F1")
    parser.add_argument(
        "optimizer", type=str, choices=["Adam", "AdamW"], help="Choose 'Adam' or 'AdamW' as the optimizer"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1, help="Dropout rate to use"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.00001, help="weight_decay value"
    )
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()

    # Print arguments for verification
    print(f"Model name: {args.model_name}")
#    print(f"Save directory: {args.save_dir}")
    print(f"Output dimension: {args.output_dim}")
    print(f"Number of blocks: {args.num_blocks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Dropout_rate: {args.dropout_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Head num for Attention: {args.n_head}")
    print(f"Threshold of metircs: {args.threshold}")
    print(f"OPTIMIZER: {args.optimizer}")

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
    data_dir = f'/nashome/uglee/EnzFormer/embed_data/{model_name}'
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.nogpu else "cpu"
    )
    k_folds = 5
    
    # Datat processing =====
    print(f"Building Dataset from {data_dir}........................")
    dataset = EmbedDataset(data_dir)
    ################ 5-Cross Validation #############

    kf = KFold(n_splits=k_folds, shuffle=True)

    fold_results = []
    train_losses = []
    val_losses = []

    early_stopping_patience = 5
    min_val_loss = np.Inf
    patience_counter = 0
    print("Training starts")
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        if fold != 0:  # Skip all folds except the first one
            continue

        print(f"FOLD {fold+1}/{k_folds}")
        print("--------------------------------")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

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

        model = ECT(
            model_name=model_name,
            output_dim=output_dim,
            num_blocks=num_blocks,
            n_head=n_head,
            dropout_rate=dropout_rate
        ).to(device)

        if output_dim == 2:
            criterion = nn.BCEWithLogitsLoss().to(device)
        elif output_dim > 2:
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            raise ValueError(
                "Invalid output_dim. It should be 2 for binary classification or greater than 2 for multi-class classification."
            )

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {args['optimizer']}")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        fold_train_losses = []
        fold_val_losses = []




        for epoch in range(num_epochs):
            avg_loss = train(model, train_dataloader, criterion, optimizer, device)
            fold_train_losses.append(avg_loss)
            val_loss, accuracy, precision, recall, f1 = validate(model, val_dataloader, criterion, device, threshold)

            fold_val_losses.append(val_loss)





            print(
                f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f},Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
            )

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
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
        if "best_model_state" in locals():
            model_save_path = os.path.join(
                save_dir, f"{model_name}_{num_blocks}_{learning_rate}_{dropout_rate}_{weight_decay}_earlystopped.pth"
            )
            torch.save(best_model_state, model_save_path)
        else:
            model_save_path = os.path.join(
                save_dir, f"{model_name}_{num_blocks}_{learning_rate}_{dropout_rate}_{weight_decay}.pth"
            )
            torch.save(model.state_dict(), model_save_path)

        train_losses.append(fold_train_losses)
        val_losses.append(fold_val_losses)










        ## Plot


        print(
            f"Cross-validation complete for fold {fold+1}. Validation Loss = {val_loss:.4f}"
        )

        # Plotting for just this fold
        plt.figure()
        plt.plot(train_losses[0], label=f"Fold {fold+1} Training Loss")
        plt.plot(val_losses[0], label=f"Fold {fold+1} Validation Loss")

        plt.title(
            f"Training and Validation Metrics for {model_name}_{num_blocks}_{learning_rate}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plot_save_path = os.path.join(
            save_dir,
            f"{model_name}_{num_blocks}_{learning_rate}_{dropout_rate}_{weight_decay}.png",
        )
        plt.savefig(plot_save_path)
        plt.close()
        print(f"Loss and metrics plot saved to {plot_save_path}")

if __name__ == "__main__":
    main()

