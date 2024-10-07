# Вывод 500 эмбеддингов:

```
{
  "limit": 500,
  "vector_name": "cryptobert_embedding"
}
```

# Вывод 500 эмбеддингов с указанием влияния новости:

```
{
  "limit": 500,
  "vector_name": "cryptobert_embedding",
  "color_by": "CryptoBERT_content_sentiment"
}
```

Доступные оценки сентимента:

```
* CryptoBERT_content_sentiment
* CryptoBERT_title_and_description_sentiment
* FinBERT_content_sentiment
* FinBERT_title_and_description_sentiment
* Llama-3-8B_content_sentiment
* Llama-3-8B_title_and_description_sentiment
* Phi-3.5-mini_content_sentiment
* Phi-3.5-mini_title_and_description_sentiment
* Mistral-7B-v0.3_content_sentiment
* Mistral-7B-v0.3_title_and_description_sentiment
```