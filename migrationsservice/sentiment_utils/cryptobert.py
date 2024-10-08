from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch


# Подгрузка cryptobert
cryptobert_name = "ElKulako/cryptobert"

cryptobert_tokenizer = AutoTokenizer.from_pretrained(cryptobert_name, use_fast=True)
cryptobert = AutoModelForSequenceClassification.from_pretrained(cryptobert_name, num_labels=3, output_hidden_states=True)
cryptobert_pipe = TextClassificationPipeline(model=cryptobert, tokenizer=cryptobert_tokenizer, max_length=64, truncation=True, padding='max_length')


def cryptobert_sentiment_calculate(news_text):
    global cryptobert_pipe

    preds = cryptobert_pipe(news_text)

    if preds[0]['label'] == "Bullish":
        preds = [{"score": preds[0]["score"], "label": "Positive"}]
    elif preds[0]['label'] == "Bearish":
        preds = [{"score": preds[0]["score"], "label": "Negative"}]

    return preds


def cryptobert_sentiment_embedding_calculate(news_text):
    global cryptobert_tokenizer, cryptobert

    inputs = cryptobert_tokenizer(
        [news_text],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = cryptobert(**inputs)
        hidden_states = outputs.hidden_states

    last_hidden_state = hidden_states[-1]
    embedding = last_hidden_state[:, 0, :][0]

    return embedding
