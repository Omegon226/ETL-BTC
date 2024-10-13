from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from newsapi import NewsApiClient
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.http.api_client import UnexpectedResponse

import os
from datetime import date, timedelta, datetime
import uuid

from sentiment_utils.cryptobert import cryptobert_sentiment_calculate, cryptobert_sentiment_embedding_calculate
from sentiment_utils.finbert import finbert_sentiment_calculate
from sentiment_utils.llm import llm_huggingface_sentiment_calculate, HUGGINGFACE_MODELS

load_dotenv()


QDRANT_URL = "http://qdrant:6333" # Прод
#QDRANT_URL = "http://localhost:8013" # Локальный дебаг
QDRANT_COLLECTION_NAME = "btc_news"
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
huggingface_hub_client = InferenceClient(api_key=os.getenv('HUGGINGFACE_KEY'))


# ОБЩИЕ ФУНКЦИИ


def create_qdrant_collection():
    global QDRANT_URL, QDRANT_COLLECTION_NAME

    q_client = QdrantClient(url=QDRANT_URL, timeout=10)

    try:
        q_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                "cryptobert_embedding": models.VectorParams(size=768, distance=models.Distance.COSINE, on_disk=True),
            },
            replication_factor=1,
            on_disk_payload=True,  # Храним payload на диске
            hnsw_config=models.HnswConfigDiff(
                ef_construct=128,
                m=32,
                full_scan_threshold = 100,
                on_disk = True
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=1.0,  # Все данные будут сохранятся в квантизированном HNSW индексе
                    always_ram=True
                )
            )
        )

        q_client.close()
    except UnexpectedResponse as error:
        print("Такая коллекция уже существует")


def create_new_columns(news_df):
    news_df["cryptobert_embedding"] = pd.Series([[] for _ in range(len(news_df))], dtype=object)

    news_df["cryptobert_content_sentiment"] = ""
    news_df["cryptobert_content_score"] = np.nan
    news_df["cryptobert_title_and_description_sentiment"] = ""
    news_df["cryptobert_title_and_description_score"] = np.nan

    news_df["finbert_content_sentiment"] = ""
    news_df["finbert_content_score"] = np.nan
    news_df["finbert_title_and_description_sentiment"] = ""
    news_df["finbert_title_and_description_score"] = np.nan

    news_df["Llama-3-8B_content_sentiment"] = ""
    news_df["Llama-3-8B_content_score"] = np.nan
    news_df["Llama-3-8B_title_and_description_sentiment"] = ""
    news_df["Llama-3-8B_title_and_description_score"] = np.nan

    news_df["Phi-3.5-mini_content_sentiment"] = ""
    news_df["Phi-3.5-mini_content_score"] = np.nan
    news_df["Phi-3.5-mini_title_and_description_sentiment"] = ""
    news_df["Phi-3.5-mini_title_and_description_score"] = np.nan

    news_df["Mistral-7B-v0.3_content_sentiment"] = ""
    news_df["Mistral-7B-v0.3_content_score"] = np.nan
    news_df["Mistral-7B-v0.3_title_and_description_sentiment"] = ""
    news_df["Mistral-7B-v0.3_title_and_description_score"] = np.nan

    return news_df

# ФУНКЦИИ ДЛЯ РАБОТЫ С API


def get_news_from_api():
    global NEWSAPI_KEY

    current_date = date.today()
    date_28_days_ago = current_date - timedelta(days=28)
    current_date_str = current_date.strftime('%Y-%m-%d')
    date_28_days_ago_str = date_28_days_ago.strftime('%Y-%m-%d')

    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    news_data = newsapi.get_everything(
        q='bitcoin',
        language='en',
        from_param=date_28_days_ago_str,
        to=current_date_str,
        sort_by='publishedAt',
        sources='crypto-coins-news, ars-technica, bloomberg, business-insider, the-next-web, the-verge, wired,' + \
                'reuters, the-washington-post, reddit-r-all, hacker-news, the-wall-street-journal, associated-press, time'
    )

    return news_data


def clean_news_from_api(news_data):
    def clean_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)

    news_df = pd.DataFrame(
        columns=["content", "title_and_description", "source", "published_at"]
    )

    for article in news_data["articles"]:
        new_row = {
            "content": clean_html(article["content"]),
            "title_and_description": clean_html(article["title"] + " " + article["description"]),
            "source": article["source"]["name"],
            "published_at": article["publishedAt"]
        }
        news_df = news_df._append(new_row, ignore_index=True)

    return news_df


def make_sentiment_analysis_for_news_from_api(news_df):
    global HUGGINGFACE_MODELS, huggingface_hub_client

    embeddings = []

    for idx in tqdm.tqdm(range(news_df.shape[0])):
        embeddings += [cryptobert_sentiment_embedding_calculate(news_df.loc[idx, "title_and_description"]).tolist()]

        cryptobert_content = cryptobert_sentiment_calculate(news_df.loc[idx, "content"])[0]
        cryptobert_title_and_description = cryptobert_sentiment_calculate(news_df.loc[idx, "title_and_description"])[0]
        news_df.loc[idx, "cryptobert_content_sentiment"] = cryptobert_content['label']
        news_df.loc[idx, "cryptobert_content_score"] = cryptobert_content['score']
        news_df.loc[idx, "cryptobert_title_and_description_sentiment"] = cryptobert_title_and_description['label']
        news_df.loc[idx, "cryptobert_title_and_description_score"] = cryptobert_title_and_description['score']

        finbert_content = finbert_sentiment_calculate(news_df.loc[idx, "content"])[0]
        finbert_title_and_description = finbert_sentiment_calculate(news_df.loc[idx, "title_and_description"])[0]
        news_df.loc[idx, "finbert_content_sentiment"] = finbert_content['label']
        news_df.loc[idx, "finbert_content_score"] = finbert_content['score']
        news_df.loc[idx, "finbert_title_and_description_sentiment"] = finbert_title_and_description['label']
        news_df.loc[idx, "finbert_title_and_description_score"] = finbert_title_and_description['score']

        llama_3_8B_content = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "content"],
            HUGGINGFACE_MODELS[0]
        )[0]
        llama_3_8B_title_and_description = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "title_and_description"],
            HUGGINGFACE_MODELS[0]
        )[0]
        news_df.loc[idx, "Llama-3-8B_content_sentiment"] = llama_3_8B_content['label']
        news_df.loc[idx, "Llama-3-8B_content_score"] = llama_3_8B_content['score']
        news_df.loc[idx, "Llama-3-8B_title_and_description_sentiment"] = llama_3_8B_title_and_description['label']
        news_df.loc[idx, "Llama-3-8B_title_and_description_score"] = llama_3_8B_title_and_description['score']

        phi_3_5_mini_content = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "content"],
            HUGGINGFACE_MODELS[2]
        )[0]
        phi_3_5_mini_title_and_description = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "title_and_description"],
            HUGGINGFACE_MODELS[2]
        )[0]
        news_df.loc[idx, "Phi-3.5-mini_content_sentiment"] = phi_3_5_mini_content['label']
        news_df.loc[idx, "Phi-3.5-mini_content_score"] = phi_3_5_mini_content['score']
        news_df.loc[idx, "Phi-3.5-mini_title_and_description_sentiment"] = phi_3_5_mini_title_and_description['label']
        news_df.loc[idx, "Phi-3.5-mini_title_and_description_score"] = phi_3_5_mini_title_and_description['score']

        mistral_7B_v0_3_content = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "content"],
            HUGGINGFACE_MODELS[3]
        )[0]
        mistral_7B_v0_3_title_and_description = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "title_and_description"],
            HUGGINGFACE_MODELS[3]
        )[0]
        news_df.loc[idx, "Mistral-7B-v0.3_content_sentiment"] = mistral_7B_v0_3_content['label']
        news_df.loc[idx, "Mistral-7B-v0.3_content_score"] = mistral_7B_v0_3_content['score']
        news_df.loc[idx, "Mistral-7B-v0.3_title_and_description_sentiment"] = mistral_7B_v0_3_title_and_description[
            'label']
        news_df.loc[idx, "Mistral-7B-v0.3_title_and_description_score"] = mistral_7B_v0_3_title_and_description['score']

    news_df["cryptobert_embedding"] = embeddings

    return news_df


def load_in_qdrant_news_from_api():
    global QDRANT_URL, QDRANT_COLLECTION_NAME

    print("Чтение данных из API")
    news_data = get_news_from_api()
    print("Обработка данных")
    news_df = clean_news_from_api(news_data)
    news_df = create_new_columns(news_df)
    #news_df = news_df.iloc[:10]
    print("Сентиментный анализ")
    news_df = make_sentiment_analysis_for_news_from_api(news_df)
    print("Сохранение информации в Qdrant")
    q_client = QdrantClient(url=QDRANT_URL, timeout=300)

    for idx in tqdm.tqdm(range(0, news_df.shape[0])):
        q_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        'published_at': news_df.loc[idx, "published_at"],
                        'source': news_df.loc[idx, "source"],

                        'CryptoBERT_content_sentiment': news_df.loc[idx, "cryptobert_content_sentiment"],
                        'CryptoBERT_content_score': float(news_df.loc[idx, "cryptobert_content_score"]),
                        'CryptoBERT_title_and_description_sentiment': news_df.loc[idx, "cryptobert_title_and_description_sentiment"],
                        'CryptoBERT_title_and_description_score': float(news_df.loc[idx, "cryptobert_title_and_description_score"]),

                        'FinBERT_content_sentiment': news_df.loc[idx, "finbert_content_sentiment"],
                        'FinBERT_content_score': float(news_df.loc[idx, "finbert_content_score"]),
                        'FinBERT_title_and_description_sentiment': news_df.loc[idx, "finbert_title_and_description_sentiment"],
                        'FinBERT_title_and_description_score': float(news_df.loc[idx, "finbert_title_and_description_score"]),

                        'Llama-3-8B_content_sentiment': news_df.loc[idx, "Llama-3-8B_content_sentiment"],
                        'Llama-3-8B_content_score': float(news_df.loc[idx, "Llama-3-8B_content_score"]),
                        'Llama-3-8B_title_and_description_sentiment': news_df.loc[idx, "Llama-3-8B_title_and_description_sentiment"],
                        'Llama-3-8B_title_and_description_score': float(news_df.loc[idx, "Llama-3-8B_title_and_description_score"]),

                        'Phi-3.5-mini_content_sentiment': news_df.loc[idx, "Phi-3.5-mini_content_sentiment"],
                        'Phi-3.5-mini_content_score': float(news_df.loc[idx, "Phi-3.5-mini_content_score"]),
                        'Phi-3.5-mini_title_and_description_sentiment': news_df.loc[idx, "Phi-3.5-mini_title_and_description_sentiment"],
                        'Phi-3.5-mini_title_and_description_score': float(news_df.loc[idx, "Phi-3.5-mini_title_and_description_score"]),

                        'Mistral-7B-v0.3_content_sentiment': news_df.loc[idx, "Mistral-7B-v0.3_content_sentiment"],
                        'Mistral-7B-v0.3_content_score': float(news_df.loc[idx, "Mistral-7B-v0.3_content_score"]),
                        'Mistral-7B-v0.3_title_and_description_sentiment': news_df.loc[idx, "Mistral-7B-v0.3_title_and_description_sentiment"],
                        'Mistral-7B-v0.3_title_and_description_score': float(news_df.loc[idx, "Mistral-7B-v0.3_title_and_description_score"]),
                    },
                    vector={
                        "cryptobert_embedding": np.array(news_df.loc[idx, "cryptobert_embedding"]).tolist(),
                    }
                ),
            ],
            wait=False
        )

    q_client.close()

# ФУНКЦИИ ДЛЯ РАБОТЫ С CSV ФАЙЛОМ


def load_news_from_csv():
    news_df = pd.read_csv("cryptonews.csv")
    news_df = news_df[news_df["subject"] == "bitcoin"].reset_index(drop=True)
    news_df = news_df.iloc[:500]

    return news_df


def make_sentiment_analysis_for_news_from_csv(news_df):
    global HUGGINGFACE_MODELS, huggingface_hub_client

    embeddings = []

    for idx in tqdm.tqdm(range(news_df.shape[0])):
        embeddings += [cryptobert_sentiment_embedding_calculate(news_df.loc[idx, "title"]).tolist()]

        cryptobert_content = cryptobert_sentiment_calculate(news_df.loc[idx, "text"])[0]
        cryptobert_title_and_description = cryptobert_sentiment_calculate(news_df.loc[idx, "title"])[0]
        news_df.loc[idx, "cryptobert_content_sentiment"] = cryptobert_content['label']
        news_df.loc[idx, "cryptobert_content_score"] = cryptobert_content['score']
        news_df.loc[idx, "cryptobert_title_and_description_sentiment"] = cryptobert_title_and_description['label']
        news_df.loc[idx, "cryptobert_title_and_description_score"] = cryptobert_title_and_description['score']

        finbert_content = finbert_sentiment_calculate(news_df.loc[idx, "text"])[0]
        finbert_title_and_description = finbert_sentiment_calculate(news_df.loc[idx, "title"])[0]
        news_df.loc[idx, "finbert_content_sentiment"] = finbert_content['label']
        news_df.loc[idx, "finbert_content_score"] = finbert_content['score']
        news_df.loc[idx, "finbert_title_and_description_sentiment"] = finbert_title_and_description['label']
        news_df.loc[idx, "finbert_title_and_description_score"] = finbert_title_and_description['score']

        llama_3_8B_content = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "text"],
            HUGGINGFACE_MODELS[0]
        )[0]
        llama_3_8B_title_and_description = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "title"],
            HUGGINGFACE_MODELS[0]
        )[0]
        news_df.loc[idx, "Llama-3-8B_content_sentiment"] = llama_3_8B_content['label']
        news_df.loc[idx, "Llama-3-8B_content_score"] = llama_3_8B_content['score']
        news_df.loc[idx, "Llama-3-8B_title_and_description_sentiment"] = llama_3_8B_title_and_description['label']
        news_df.loc[idx, "Llama-3-8B_title_and_description_score"] = llama_3_8B_title_and_description['score']

        phi_3_5_mini_content = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "text"],
            HUGGINGFACE_MODELS[2]
        )[0]
        phi_3_5_mini_title_and_description = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "title"],
            HUGGINGFACE_MODELS[2]
        )[0]
        news_df.loc[idx, "Phi-3.5-mini_content_sentiment"] = phi_3_5_mini_content['label']
        news_df.loc[idx, "Phi-3.5-mini_content_score"] = phi_3_5_mini_content['score']
        news_df.loc[idx, "Phi-3.5-mini_title_and_description_sentiment"] = phi_3_5_mini_title_and_description['label']
        news_df.loc[idx, "Phi-3.5-mini_title_and_description_score"] = phi_3_5_mini_title_and_description['score']

        mistral_7B_v0_3_content = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "text"],
            HUGGINGFACE_MODELS[3]
        )[0]
        mistral_7B_v0_3_title_and_description = llm_huggingface_sentiment_calculate(
            huggingface_hub_client,
            news_df.loc[idx, "title"],
            HUGGINGFACE_MODELS[3]
        )[0]
        news_df.loc[idx, "Mistral-7B-v0.3_content_sentiment"] = mistral_7B_v0_3_content['label']
        news_df.loc[idx, "Mistral-7B-v0.3_content_score"] = mistral_7B_v0_3_content['score']
        news_df.loc[idx, "Mistral-7B-v0.3_title_and_description_sentiment"] = mistral_7B_v0_3_title_and_description['label']
        news_df.loc[idx, "Mistral-7B-v0.3_title_and_description_score"] = mistral_7B_v0_3_title_and_description['score']

    news_df["cryptobert_embedding"] = embeddings

    return news_df


def load_in_qdrant_news_from_csv():
    global QDRANT_URL, QDRANT_COLLECTION_NAME

    print("Чтение данных из CSV файла")
    news_df = load_news_from_csv()
    print("Обработка данных")
    news_df = create_new_columns(news_df)
    #news_df = news_df.iloc[:10]
    print("Сентиментный анализ")
    news_df = make_sentiment_analysis_for_news_from_csv(news_df)
    print("Сохранение информации в Qdrant")
    q_client = QdrantClient(url=QDRANT_URL, timeout=300)

    for idx in tqdm.tqdm(range(0, news_df.shape[0])):
        q_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        'published_at': news_df.loc[idx, "date"],
                        'source': news_df.loc[idx, "source"],

                        'CryptoBERT_content_sentiment': news_df.loc[idx, "cryptobert_content_sentiment"],
                        'CryptoBERT_content_score': float(news_df.loc[idx, "cryptobert_content_score"]),
                        'CryptoBERT_title_and_description_sentiment': news_df.loc[idx, "cryptobert_title_and_description_sentiment"],
                        'CryptoBERT_title_and_description_score': float(news_df.loc[idx, "cryptobert_title_and_description_score"]),

                        'FinBERT_content_sentiment': news_df.loc[idx, "finbert_content_sentiment"],
                        'FinBERT_content_score': float(news_df.loc[idx, "finbert_content_score"]),
                        'FinBERT_title_and_description_sentiment': news_df.loc[idx, "finbert_title_and_description_sentiment"],
                        'FinBERT_title_and_description_score': float(news_df.loc[idx, "finbert_title_and_description_score"]),

                        'Llama-3-8B_content_sentiment': news_df.loc[idx, "Llama-3-8B_content_sentiment"],
                        'Llama-3-8B_content_score': float(news_df.loc[idx, "Llama-3-8B_content_score"]),
                        'Llama-3-8B_title_and_description_sentiment': news_df.loc[idx, "Llama-3-8B_title_and_description_sentiment"],
                        'Llama-3-8B_title_and_description_score': float(news_df.loc[idx, "Llama-3-8B_title_and_description_score"]),

                        'Phi-3.5-mini_content_sentiment': news_df.loc[idx, "Phi-3.5-mini_content_sentiment"],
                        'Phi-3.5-mini_content_score': float(news_df.loc[idx, "Phi-3.5-mini_content_score"]),
                        'Phi-3.5-mini_title_and_description_sentiment': news_df.loc[idx, "Phi-3.5-mini_title_and_description_sentiment"],
                        'Phi-3.5-mini_title_and_description_score': float(news_df.loc[idx, "Phi-3.5-mini_title_and_description_score"]),

                        'Mistral-7B-v0.3_content_sentiment': news_df.loc[idx, "Mistral-7B-v0.3_content_sentiment"],
                        'Mistral-7B-v0.3_content_score': float(news_df.loc[idx, "Mistral-7B-v0.3_content_score"]),
                        'Mistral-7B-v0.3_title_and_description_sentiment': news_df.loc[idx, "Mistral-7B-v0.3_title_and_description_sentiment"],
                        'Mistral-7B-v0.3_title_and_description_score': float(news_df.loc[idx, "Mistral-7B-v0.3_title_and_description_score"]),
                    },
                    vector={
                        "cryptobert_embedding": np.array(news_df.loc[idx, "cryptobert_embedding"]).tolist(),
                    }
                ),
            ],
            wait=False
        )

    q_client.close()

# ФУНКЦИЯ ИНИЦИАЛИЗАЦИИ


def init_qdrant():
    print("Создание Qdrant коллекции")
    create_qdrant_collection()
    print("Анализ новостей из API")
    load_in_qdrant_news_from_api()
    print("Анализ новостей из CSV файла")
    load_in_qdrant_news_from_csv()
