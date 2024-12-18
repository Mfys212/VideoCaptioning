import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm 
import nltk
nltk.download('wordnet')

class CalculateMetrics(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, true, pred):
        true = [item.decode('utf-8') for item in true.numpy()]
        pred = [item.decode('utf-8') for item in pred.numpy()]
        true = [text.lower() for text in true]
        pred = [text.lower() for text in pred]
        BLUE, ROUGE1, ROUGE2, ROUGE3, METEOR = [], [], [], [], []
        for i, j in tqdm(zip(true, pred), desc="Calculate metrics"):
            try:
                bleu, rouge1, rouge2, rouge3, meteor = self.calculate_metrics(i , j)
            except:
                continue
            BLUE.append(bleu)
            ROUGE1.append(rouge1)
            ROUGE2.append(rouge2)
            ROUGE3.append(rouge3)
            METEOR.append(meteor)
                
        return {
          "BLEU": np.array(BLUE).mean(),
          "ROUGE-1": np.array(ROUGE1).mean(),
          "ROUGE-2": np.array(ROUGE2).mean(),
          "ROUGE-L": np.array(ROUGE3).mean(),
          "METEOR": np.array(METEOR).mean()
        }

    def calculate_metrics(self, reference, hypothesis, use_stemmer=True):
        # references = [ref.split() for ref in reference]
        references = [reference.split()]
        hypothesis_tokens = hypothesis.split()
        smoothing_function = SmoothingFunction().method4
        bleu_score = sentence_bleu(references, hypothesis_tokens, smoothing_function=smoothing_function)
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        rouge_scores = rouge_scorer_obj.score(reference[0], hypothesis)
        meteor = meteor_score(references, hypothesis_tokens)
        return bleu_score, rouge_scores["rouge1"].fmeasure, rouge_scores["rouge2"].fmeasure, rouge_scores["rougeL"].fmeasure, meteor
