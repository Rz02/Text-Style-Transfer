import evaluate

def compute_bleu(predictions, references):
    """
    Compute BLEU score using sacreBLEU.
    Args:
        predictions (list[str]): Generated sentences.
        references (list[str]): Reference sentences.
    Returns:
        dict: BLEU score.
    """
    bleu = evaluate.load("sacrebleu")
    results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    return results

def compute_bert_score(predictions, references):
    """
    Compute BERTScore.
    """
    bert_score = evaluate.load("bertscore")
    results = bert_score.compute(predictions=predictions, references=references, lang="en")
    return results

def compute_meteor(predictions, references):
    """
    Compute METEOR score.
    """
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predictions, references=references)
    return results

def compute_all_metrics(predictions, references):
    """
    Compute a suite of evaluation metrics (BLEU, BERTScore, METEOR)
    Args:
        predictions (list[str]): Generated sentences.
        references (list[str]): Reference sentences.
    Returns:
        dict: A dictionary containing all computed metrics.
    """
    metrics = {
        "bleu": compute_bleu(predictions, references),
        "bert_score": compute_bert_score(predictions, references),
        "meteor": compute_meteor(predictions, references)
    }
    return metrics

if __name__ == "__main__":
    preds = ["This is a detoxified sentence.", "Another detoxified sentence."]
    refs = ["This is a detoxified sentence.", "Another detoxified sentence."]
    
    all_metrics = compute_all_metrics(preds, refs)
    print(all_metrics)