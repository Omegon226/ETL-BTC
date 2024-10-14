from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.exceptions import AirflowSkipException

from qdrant_client import QdrantClient, models
import uuid
from newsapi import NewsApiClient
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import os
from bs4 import BeautifulSoup
import logging
import pandas as pd


load_dotenv(dotenv_path=Path('/home/airflow/.env'))

NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
SEMANTIC_ANALYSIS_URL = 'http://sentiment_analysis_service:8000/api/make_analysis/full/'
QDRANT_URL = 'http://qdrant:6333'
QDRANT_COLLECTION = 'btc_news'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 5,  
    'retry_delay': timedelta(minutes=1),  
} 


def fetch_news(**kwargs):
    logging.info(NEWSAPI_KEY)
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=24+1)
    
    from_time = one_hour_ago.strftime('%Y-%m-%dT%H:%M:%S')
    to_time = now.strftime('%Y-%m-%dT%H:%M:%S')
    
    logging.info("Получение новостей")
    all_articles = newsapi.get_everything(
        q='bitcoin', 
        language='en',
        from_param=from_time,
        to=to_time,
        sort_by='publishedAt',
        sources='crypto-coins-news, ars-technica, bloomberg, business-insider, the-next-web, the-verge, wired,' +\
        'reuters, the-washington-post, reddit-r-all, hacker-news, the-wall-street-journal, associated-press, time'
    )
    
    articles = all_articles["articles"]

    if len(articles) == 0:
        raise AirflowSkipException("Нет новостей за последний час.")
        
    # Сохранение данных в XCom
    logging.info("Сохранение новостей в XCom")
    kwargs['ti'].xcom_push(key='raw_news', value=articles)


def process_data(**kwargs):
    def clean_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    
    logging.info("Загрузка новостей из XCom")
    articles = kwargs['ti'].xcom_pull(key='raw_news', task_ids='fetch_news')
    if len(articles) == 0:
        raise ValueError("Нет данных для обработки.")
    
    logging.info("Обработка новостей")
    processed_news = []

    for article in articles:
        processed_news += [{
            "content": clean_html(article["content"]),
            "title_and_description": clean_html(article["title"] + " " + article["description"]),
            "source": article["source"]["name"],
            "published_at": article["publishedAt"]
        }]

    # Сохранение обработанных данных в XCom
    logging.info("Сохранение обработанных новостей в XCom")
    kwargs['ti'].xcom_push(key='processed_data', value=processed_news)


def semantic_analysis(**kwargs):
    logging.info("Загрузка обработанных новостей из XCom")
    processed_news = kwargs['ti'].xcom_pull(key='processed_data', task_ids='process_data')
    if len(processed_news) == 0:
        raise ValueError("Нет обработанных данных для семантического анализа.")
    
    logging.info("Семантический анализ обработанных новостей")
    analyzed_data = []
    headers = {'Content-Type': 'application/json'}
    
    for news in processed_news:     
        try:
            data = {
                "title": news["title_and_description"],
                "text": news["content"],
            }
            response = requests.post(SEMANTIC_ANALYSIS_URL, params=data, headers=headers)
            response.raise_for_status()
            analysis_result = response.json()  # Ожидается, что API возвращает dict
            analysis_result["source"] = news["source"]
            analysis_result["published_at"] = news["published_at"]
            analyzed_data.append(analysis_result)
        except requests.exceptions.RequestException as e:
            # Логирование ошибки и продолжение
            print(f"Ошибка при семантическом анализе статьи: {e}")
            continue
    
    if not analyzed_data:
        raise ValueError("Семантический анализ не дал результатов.")
    
    # Сохранение результатов в XCom
    logging.info("Сохранение результатов семантического анализа в XCom")
    kwargs['ti'].xcom_push(key='analyzed_data', value=analyzed_data)


def save_to_qdrant(**kwargs):
    logging.info("Загрузка результатов семантического анализа из XCom")
    analyzed_data = kwargs['ti'].xcom_pull(key='analyzed_data', task_ids='semantic_analysis')
    if not analyzed_data:
        raise ValueError("Нет данных для сохранения в Qdrant.")


    logging.info("Сохранение результатов семантического анализа в Qdrant")
    client = QdrantClient(url=QDRANT_URL, timeout=300)
    
    # Проверка существования коллекции
    try:
        client.get_collection(collection_name=QDRANT_COLLECTION)
    except Exception:
        raise Exception(f"Коллекции {QDRANT_COLLECTION} не существует")
    
    # Вставка данных
    for idx in range(len(analyzed_data)):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        'published_at': analyzed_data[idx]["published_at"],
                        'source': analyzed_data[idx]["source"],

                        'CryptoBERT_content_sentiment': analyzed_data[idx]["cryptobert_content_sentiment"],
                        'CryptoBERT_content_score': float(analyzed_data[idx]["cryptobert_content_score"]),
                        'CryptoBERT_title_and_description_sentiment': analyzed_data[idx]["cryptobert_title_and_description_sentiment"],
                        'CryptoBERT_title_and_description_score': float(analyzed_data[idx]["cryptobert_title_and_description_score"]),

                        'FinBERT_content_sentiment': analyzed_data[idx]["finbert_content_sentiment"],
                        'FinBERT_content_score': float(analyzed_data[idx]["finbert_content_score"]),
                        'FinBERT_title_and_description_sentiment': analyzed_data[idx]["finbert_title_and_description_sentiment"],
                        'FinBERT_title_and_description_score': float(analyzed_data[idx]["finbert_title_and_description_score"]),

                        'Llama-3-8B_content_sentiment': analyzed_data[idx]["llama_3_8b_content_sentiment"],
                        'Llama-3-8B_content_score': float(analyzed_data[idx]["llama_3_8b_content_score"]),
                        'Llama-3-8B_title_and_description_sentiment': analyzed_data[idx]["llama_3_8b_title_and_description_sentiment"],
                        'Llama-3-8B_title_and_description_score': float(analyzed_data[idx]["llama_3_8b_title_and_description_score"]),

                        'Phi-3.5-mini_content_sentiment': analyzed_data[idx]["phi_3_5_mini_content_sentiment"],
                        'Phi-3.5-mini_content_score': float(analyzed_data[idx]["phi_3_5_mini_content_score"]),
                        'Phi-3.5-mini_title_and_description_sentiment': analyzed_data[idx]["phi_3_5_mini_title_and_description_sentiment"],
                        'Phi-3.5-mini_title_and_description_score': float(analyzed_data[idx]["phi_3_5_mini_title_and_description_score"]),

                        'Mistral-7B-v0.3_content_sentiment': analyzed_data[idx]["mistral_7b_v0_3_content_sentiment"],
                        'Mistral-7B-v0.3_content_score': float(analyzed_data[idx]["mistral_7b_v0_3_content_score"]),
                        'Mistral-7B-v0.3_title_and_description_sentiment': analyzed_data[idx]["mistral_7b_v0_3_title_and_description_sentiment"],
                        'Mistral-7B-v0.3_title_and_description_score': float(analyzed_data[idx]["mistral_7b_v0_3_title_and_description_score"]),
                    },
                    vector={
                        "cryptobert_embedding": analyzed_data[idx]["embedding"],
                    }
                ),
            ],
            wait=False
        )

    client.close()


with DAG(
    'news_etl_dag',
    default_args=default_args,
    description='ETL процесс для получения новостей за последний час, их анализа и сохранения в Qdrant',
    schedule_interval=timedelta(hours=1),  # Запуск каждый час
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    fetch_news_task = PythonOperator(
        task_id='fetch_news',
        python_callable=fetch_news,
        provide_context=True
    )

    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        provide_context=True
    )

    semantic_analysis_task = PythonOperator(
        task_id='semantic_analysis',
        python_callable=semantic_analysis,
        provide_context=True
    )

    save_to_qdrant_task = PythonOperator(
        task_id='save_to_qdrant',
        python_callable=save_to_qdrant,
        provide_context=True
    )

    # Определение последовательности задач
    fetch_news_task >> process_data_task >> semantic_analysis_task >> save_to_qdrant_task