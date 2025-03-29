import pandas as pd

def split_tsv(input_file, train_file, eval_file, eval_size=500, random_seed=42):
    """
    Splits a TSV file into two files: one for training and one for evaluation.
    
    Args:
        input_file (str): Path to the input TSV file.
        train_file (str): Path to save the training TSV file.
        eval_file (str): Path to save the evaluation TSV file.
        eval_size (int): Number of samples to randomly select for evaluation.
        random_seed (int): Random seed for reproducibility.
    """
    df = pd.read_csv(input_file, sep="\t")
    
    eval_df = df.sample(n=eval_size, random_state=random_seed)
    
    train_df = df.drop(eval_df.index)
    
    train_df.to_csv(train_file, sep="\t", index=False)
    eval_df.to_csv(eval_file, sep="\t", index=False)
    
    print(f"Saved {len(train_df)} training samples to {train_file}")
    print(f"Saved {len(eval_df)} evaluation samples to {eval_file}")

if __name__ == "__main__":
    input_file = "Data/paradetox.tsv"
    train_file = "Data/paradetox_train.tsv"
    eval_file = "Data/paradetox_eval.tsv"
    split_tsv(input_file, train_file, eval_file)
