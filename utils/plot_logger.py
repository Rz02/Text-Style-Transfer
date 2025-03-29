import logging
import os
import matplotlib.pyplot as plt

def setup_logger(log_file: str = "training.log"):
    """
    Set up a logger that writes to both console and a log file.

    Args:
        log_file (str): The file path to write logs.
    
    Returns:
        logger: A Python logger instance.
    """
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def log_metrics(logger, step: int, metrics: dict):
    """
    Log evaluation metrics to both console and a log file.
    
    Args:
        logger: The logger instance.
        step (int): The current training step or epoch.
        metrics (dict): A dictionary containing metric names and their values.
    """
    message = f"Step {step}: " + ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    logger.info(message)

def plot_metric(metric_values: list, metric_name: str, save_dir: str = None):
    """
    Plot the progression of a single metric over training steps or epochs.
    
    Args:
        metric_values (list[float]): List of metric values recorded over training.
        metric_name (str): Name of the metric (e.g., "BLEU", "BERTScore").
        save_dir (str, optional): Directory where the plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(metric_values, marker='o', linestyle='-', color='b')
    plt.title(f"{metric_name} over Training Steps")
    plt.xlabel("Step/Epoch")
    plt.ylabel(metric_name)
    plt.grid(True)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric_name}_plot.png")
        plt.savefig(save_path)
        print(f"Saved {metric_name} plot to {save_path}")
    plt.show()

def plot_losses(train_losses: list, eval_losses: list, save_dir: str = None):
    """
    Plot training and evaluation losses over epochs.

    Args:
        train_losses (list[float]): List of training loss values recorded over epochs.
        eval_losses (list[float]): List of evaluation loss values recorded over epochs.
        save_dir (str, optional): Directory where the loss plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', label="Training Loss", color='r')
    plt.plot(epochs, eval_losses, marker='o', linestyle='-', label="Evaluation Loss", color='g')
    plt.title("Training and Evaluation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_plot.png")
        plt.savefig(save_path)
        print(f"Saved loss plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    logger = setup_logger("example_training.log")

    bleu_scores = [0.25, 0.30, 0.35, 0.37, 0.40]
    bert_scores = [0.85, 0.87, 0.88, 0.89, 0.90]
    metrics = {"BLEU": bleu_scores[-1], "BERTScore": bert_scores[-1]}
    log_metrics(logger, step=5, metrics=metrics)
    
    plot_metric(bleu_scores, "BLEU", save_dir="plots")
    plot_metric(bert_scores, "BERTScore", save_dir="plots")
    
    train_losses = [2.3, 2.1, 1.9, 1.8, 1.7]
    eval_losses = [2.4, 2.2, 2.0, 1.95, 1.85]
    
    plot_losses(train_losses, eval_losses, save_dir="plots")
