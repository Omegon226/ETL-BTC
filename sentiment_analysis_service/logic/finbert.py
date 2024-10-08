from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer


# Подгрузка finbert
finbert_name = "yiyanghkust/finbert-tone"

finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_name, use_fast=True)
finbert = AutoModelForSequenceClassification.from_pretrained(finbert_name, num_labels = 3)
finbert_pipe = TextClassificationPipeline(model=finbert, tokenizer=finbert_tokenizer, max_length=64, truncation=True, padding='max_length')


def finbert_sentiment_calculate(news_text):
    global finbert_pipe

    preds = finbert_pipe(news_text)

    return (preds)
