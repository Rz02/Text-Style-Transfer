from datasets import load_dataset as hf_load_dataset
import os

def read_dataset(tsv_path: str):
    """
    Reads a TSV file containing toxic and neutral sentences.
    
    The TSV file should have a header with columns:
        toxic    neutral1    neutral2    neutral3
    
    Some rows may have empty values for neutral2 and neutral3.
    
    Args:
        tsv_path (str): Path to the TSV file.
    
    Returns:
        dataset: A Hugging Face Dataset loaded from the TSV file.
    """
    dataset = hf_load_dataset("csv", data_files=tsv_path, delimiter="\t", split="train")
    return dataset

if __name__ == "__main__":
    dataset = read_dataset('Data\paradetox.tsv')
    print(dataset[0])