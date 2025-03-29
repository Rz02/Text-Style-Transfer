from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

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

class DetoxificationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int = 512):
        """
        Initialize the DetoxificationDataset.

        Args:
            hf_dataset: A Hugging Face dataset containing the columns 'toxic', 'neutral1', 'neutral2', 'neutral3'.
            tokenizer (T5Tokenizer): The tokenizer to use for tokenizing texts.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        toxic_text = item["toxic"]
        neutral_text = item.get("neutral1") or item.get("neutral2") or item.get("neutral3")
        prompt = "detoxify text: "
        toxic_text_with_prompt = prompt + toxic_text

        input_encodings = self.tokenizer(
            toxic_text_with_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_encodings = self.tokenizer(
            toxic_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encodings = self.tokenizer(
            neutral_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = input_encodings["input_ids"].squeeze()
        attention_mask = input_encodings["attention_mask"].squeeze()
        labels = target_encodings["input_ids"].squeeze()
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "toxic_text": toxic_text  # original toxic text for cycle loss if needed
        }

def create_dataloader(tsv_path: str, tokenizer, batch_size: int = 16, max_length: int = 512, shuffle: bool = True):
    """
    Creates a DataLoader for the detoxification dataset.

    Args:
        tsv_path (str): Path to the TSV file.
        tokenizer (T5Tokenizer): The tokenizer to use.
        batch_size (int): Number of samples per batch.
        max_length (int): Maximum sequence length for tokenization.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    hf_dataset = read_dataset(tsv_path)
    dataset = DetoxificationDataset(hf_dataset, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataloader = create_dataloader("Data/paradetox.tsv", tokenizer, batch_size=8)

    # Iterate over one batch to test the loader
    for batch in dataloader:
        print("Input IDs:", batch["input_ids"].shape)
        print("Attention Mask:", batch["attention_mask"].shape)
        print("Labels:", batch["labels"].shape)
        break