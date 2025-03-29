import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

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

def compute_content_preservation(original_texts, generated_texts, embedding_model=None):
    """
    Computes content preservation as the average cosine similarity between embeddings of
    the original and generated sentences.
    
    Args:
        original_texts (list[str]): The original sentences.
        generated_texts (list[str]): The generated sentences.
        embedding_model: A SentenceTransformer model for obtaining embeddings. If None,
            uses "paraphrase-MiniLM-L6-v2".
    
    Returns:
        float: Average cosine similarity.
    """
    if embedding_model is None:
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    
    orig_embeddings = embedding_model.encode(original_texts, convert_to_tensor=True)
    gen_embeddings = embedding_model.encode(generated_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(orig_embeddings, gen_embeddings)
    diag_sim = cosine_scores.diag().cpu().numpy()
    return np.mean(diag_sim)

def compute_fluency(sentences, fluency_pipeline=None):
    """
    Computes fluency as the percentage of sentences classified as fluent by a RoBERTa-based classifier.
    
    Args:
        sentences (list[str]): List of generated sentences.
        fluency_pipeline: A transformers pipeline for text classification. If None, uses
            the "textattack/roberta-base-CoLA" model.
    
    Returns:
        float: Fluency percentage (between 0 and 1).
    """
    if fluency_pipeline is None:
        fluency_pipeline = pipeline("text-classification", model="textattack/roberta-base-CoLA")
    
    results = fluency_pipeline(sentences)
    
    fluent_count = 0
    for result in results:
        label = result.get("label", "")
        if label in ["LABEL_1", "Acceptable"]:
            fluent_count += 1
    return fluent_count / len(sentences) if sentences else 0.0

def compute_all_metrics(predictions, references, original_texts=None):
    """
    Compute a suite of evaluation metrics including BLEU, BERTScore, METEOR, Content Preservation (SIM), and Fluency (FL).
    
    Args:
        predictions (list[str]): Generated detoxified sentences.
        references (list[str]): Reference detoxified sentences.
        original_texts (list[str], optional): Original toxic sentences used to compute content preservation.
                                               If None, content preservation is skipped.
    
    Returns:
        dict: A dictionary containing all computed metrics.
    """
    metrics = {}
    metrics["bleu"] = compute_bleu(predictions, references)
    # metrics["bert_score"] = compute_bert_score(predictions, references)
    metrics["meteor"] = compute_meteor(predictions, references)
    # if original_texts is not None:
    #     metrics["content_preservation"] = compute_content_preservation(original_texts, predictions)
    # else:
    #     metrics["content_preservation"] = None
    # metrics["fluency"] = compute_fluency(predictions)
    return metrics

if __name__ == "__main__":
    preds = ["This is a detoxified sentence.", "Another detoxified sentence."]
    refs = ["This is a detoxified sentence.", "Another detoxified sentence."]
    
    original_texts = ["This is a toxic sentence.", "Another toxic sentence."]
    
    all_metrics = compute_all_metrics(preds, refs, original_texts=original_texts)
    
    print("Evaluation Metrics:")
    print("BLEU:", all_metrics["bleu"])
    print("BERTScore:", all_metrics["bert_score"])
    print("METEOR:", all_metrics["meteor"])
    if all_metrics["content_preservation"] is not None:
        print("Content Preservation (SIM):", all_metrics["content_preservation"])
    print("Fluency (FL):", all_metrics["fluency"])
