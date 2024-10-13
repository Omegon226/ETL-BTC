import streamlit as st
import os
import requests
import json
from qdrant_client import QdrantClient
import plotly.graph_objs as go

from styles.page_style import set_page_config_wide, disable_header_and_footer

# Настройка страницы
set_page_config_wide()
disable_header_and_footer()

SEMANTIC_ANALYSIS_URL = 'http://sentiment_analysis_service:8000/api/make_analysis/full/'
#SEMANTIC_ANALYSIS_URL = 'http://localhost:8001/api/make_analysis/full/'
QDRANT_URL = 'http://qdrant:6333'
#QDRANT_URL = 'http://localhost:8013'
QDRANT_COLLECTION = 'btc_news'


def update_sentiment(model_sentiment):
    global sentiment

    if model_sentiment == "Positive":
        sentiment["Positive"] += 1
    elif model_sentiment == "Negative":
        sentiment["Negative"] += 1
    elif model_sentiment == "Neutral":
        sentiment["Neutral"] += 1



st.markdown(
    """
    # Сентиментный анализ новости
    """
)

custom_title = st.text_input("Заголовок")
custom_text = st.text_area("Текст")

if st.button("Отправить на анализ"):
    if not custom_title or not custom_text:
        st.warning("Пожалуйста, заполните оба поля.")
    else:
        # Подготавливаем данные для отправки
        data = {
            'title': custom_title,
            'text': custom_text
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(SEMANTIC_ANALYSIS_URL, params=data, headers=headers)

        try:
            # Отправляем POST-запрос с JSON-данными
            response = requests.post(SEMANTIC_ANALYSIS_URL, params=data, headers=headers)
            result = response.json()  # Предполагаем, что ответ в формате JSON
            st.success("Анализ выполнен успешно!")

            st.markdown(
                f"""
                ## Результат анализа
                
                * Оценка CryptoBERT на основе содержания новости {result["cryptobert_content_sentiment"]} с уверенностью {result["cryptobert_content_score"]}
                * Оценка CryptoBERT на основе заголовка новости {result["cryptobert_title_and_description_sentiment"]} с уверенностью {result["cryptobert_title_and_description_score"]}
                * Оценка FinBERT на основе содержания новости {result["finbert_content_sentiment"]} с уверенностью {result["finbert_content_score"]}
                * Оценка FinBERT на основе заголовка новости {result["finbert_title_and_description_sentiment"]} с уверенностью {result["finbert_title_and_description_score"]}
                * Оценка Llama 3 8B на основе содержания новости {result["llama_3_8b_content_sentiment"]} с уверенностью {result["llama_3_8b_content_score"]}
                * Оценка Llama 3 8B на основе заголовка новости {result["llama_3_8b_title_and_description_sentiment"]} с уверенностью {result["llama_3_8b_title_and_description_score"]}
                * Оценка Phi 3.5 mini на основе содержания новости {result["phi_3_5_mini_content_sentiment"]} с уверенностью {result["phi_3_5_mini_content_score"]}
                * Оценка Phi 3.5 mini на основе заголовка новости {result["phi_3_5_mini_title_and_description_sentiment"]} с уверенностью {result["phi_3_5_mini_title_and_description_score"]}
                * Оценка Mistral 7B v0.3 на основе содержания новости {result["mistral_7b_v0_3_content_sentiment"]} с уверенностью {result["mistral_7b_v0_3_content_score"]}
                * Оценка Mistral 7B v0.3 на основе заголовка новости {result["mistral_7b_v0_3_title_and_description_sentiment"]} с уверенностью {result["mistral_7b_v0_3_title_and_description_score"]}                
                """
            )

            q_client = QdrantClient(url=QDRANT_URL, timeout=100)

            # Данные для визуализации
            qdrant_result = q_client.query_points(
                collection_name="btc_news",
                query=result["embedding"],
                using="cryptobert_embedding",
                with_payload=True,
            )
            qdrant_result = list(qdrant_result)[0][1]

            sentiment = {"Positive": 0, "Negative": 0, "Neutral": 0}

            for idx in range(len(qdrant_result)):
                query_result = qdrant_result[idx].__dict__["payload"]

                update_sentiment(query_result["CryptoBERT_content_sentiment"])
                update_sentiment(query_result["CryptoBERT_title_and_description_sentiment"])
                update_sentiment(query_result["FinBERT_content_sentiment"])
                update_sentiment(query_result["FinBERT_title_and_description_sentiment"])
                update_sentiment(query_result["Llama-3-8B_content_sentiment"])
                update_sentiment(query_result["Llama-3-8B_title_and_description_sentiment"])
                update_sentiment(query_result["Mistral-7B-v0.3_content_sentiment"])
                update_sentiment(query_result["Mistral-7B-v0.3_title_and_description_sentiment"])
                update_sentiment(query_result["Phi-3.5-mini_content_sentiment"])
                update_sentiment(query_result["Phi-3.5-mini_title_and_description_sentiment"])

            q_client.close()

            # Создаем столбчатую диаграмму с помощью Plotly
            fig = go.Figure(data=[
                go.Bar(name='Sentiment', x=list(sentiment.keys()), y=list(sentiment.values()))
            ])

            # Добавляем заголовок и подписи к осям
            fig.update_layout(
                title='Sentiment Analysis',
                xaxis_title='Sentiment',
                yaxis_title='Count',
                yaxis=dict(range=[0, max(sentiment.values()) + 10]),  # Диапазон Y с небольшим отступом
            )

            st.title('Визуализация поиска по БД похожих новостей')
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при подключении к серверу: {e}")