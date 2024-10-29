import os
from torch.utils.data import Dataset


class FastaDataset(Dataset):
    """
    Fasta Files is a list that contains the file paths to the Fata files.
    """
    def __init__(self, fasta_files):
        self.original_fasta_files = fasta_files
        self.annotations = []
        self.fasta_files = []

        for fasta_file in fasta_files:
            modified_file = self.remove_top_line_and_save(fasta_file)
            self.fasta_files.append(modified_file)
            annotation = self.load_annotation(fasta_file)
            # If there are multiple annotations, store them as a list of integers
            annotation_list = [int(ann) for ann in annotation.split(",")]
            self.annotations.append(annotation_list)

        # Mapping annotations to indices
        unique_annotations = sorted(
            set(sum(self.annotations, []))
        )  # Flatten the list of lists
        self.annotation_to_index = {
            val: idx for idx, val in enumerate(unique_annotations)
        }
        # Replace annotations with their corresponding indices
        self.annotations = [
            [self.annotation_to_index[ann] for ann in annotation_list]
            for annotation_list in self.annotations
        ]

    def remove_top_line_and_save(self, fasta_file):
        base, ext = os.path.splitext(fasta_file)
        new_file = f"{base}_temp{ext}"

        with open(fasta_file, "r") as infile, open(new_file, "w") as outfile:
            next(infile)
            for line in infile:
                outfile.write(line)

        return new_file

    def load_annotation(self, fasta_file):
        with open(fasta_file, "r") as file:
            annotation = file.readline().strip()
        return annotation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        fasta_file = self.fasta_files[idx]

        return annotation, fasta_file
