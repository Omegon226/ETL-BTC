from huggingface_hub import InferenceClient
import os

from logic.cryptobert import cryptobert_sentiment_calculate, cryptobert_sentiment_embedding_calculate
from logic.finbert import finbert_sentiment_calculate
from logic.llm import llm_huggingface_sentiment_calculate, HUGGINGFACE_MODELS
from models.make_analysis_response import MakeAnalysisResponse

huggingface_hub_client = InferenceClient(api_key=os.getenv('HUGGINGFACE_KEY'))


def make_sentiment_analysis(title, text):
    embedding = cryptobert_sentiment_embedding_calculate(title).tolist()

    cryptobert_text = cryptobert_sentiment_calculate(text)[0]
    cryptobert_title = cryptobert_sentiment_calculate(title)[0]
    cryptobert_content_sentiment = cryptobert_text['label']
    cryptobert_content_score = cryptobert_text['score']
    cryptobert_title_and_description_sentiment = cryptobert_title['label']
    cryptobert_title_and_description_score = cryptobert_title['score']

    finbert_text = finbert_sentiment_calculate(text)[0]
    finbert_title = finbert_sentiment_calculate(title)[0]
    finbert_content_sentiment = finbert_text['label']
    finbert_content_score = finbert_text['score']
    finbert_title_and_description_sentiment = finbert_title['label']
    finbert_title_and_description_score = finbert_title['score']

    llama_3_8b_text = llm_huggingface_sentiment_calculate(
        huggingface_hub_client,
        text,
        HUGGINGFACE_MODELS[0]
    )[0]
    llama_3_8b_title = llm_huggingface_sentiment_calculate(
        huggingface_hub_client,
        title,
        HUGGINGFACE_MODELS[0]
    )[0]
    llama_3_8b_content_sentiment = llama_3_8b_text['label']
    llama_3_8b_content_score = llama_3_8b_text['score']
    llama_3_8b_title_and_description_sentiment = llama_3_8b_title['label']
    llama_3_8b_title_and_description_score = llama_3_8b_title['score']

    phi_3_5_mini_text = llm_huggingface_sentiment_calculate(
        huggingface_hub_client,
        text,
        HUGGINGFACE_MODELS[2]
    )[0]
    phi_3_5_mini_title = llm_huggingface_sentiment_calculate(
        huggingface_hub_client,
        title,
        HUGGINGFACE_MODELS[2]
    )[0]
    phi_3_5_mini_content_sentiment = phi_3_5_mini_text['label']
    phi_3_5_mini_content_score = phi_3_5_mini_text['score']
    phi_3_5_mini_title_and_description_sentiment = phi_3_5_mini_title['label']
    phi_3_5_mini_title_and_description_score = phi_3_5_mini_title['score']

    mistral_7b_v0_3_text = llm_huggingface_sentiment_calculate(
        huggingface_hub_client,
        text,
        HUGGINGFACE_MODELS[3]
    )[0]
    mistral_7b_v0_3_title = llm_huggingface_sentiment_calculate(
        huggingface_hub_client,
        title,
        HUGGINGFACE_MODELS[3]
    )[0]
    mistral_7b_v0_3_content_sentiment = mistral_7b_v0_3_text['label']
    mistral_7b_v0_3_content_score = mistral_7b_v0_3_text['score']
    mistral_7b_v0_3_title_and_description_sentiment = mistral_7b_v0_3_title['label']
    mistral_7b_v0_3_title_and_description_score = mistral_7b_v0_3_title['score']

    result = MakeAnalysisResponse(
        embedding=embedding,
        cryptobert_content_sentiment=cryptobert_content_sentiment,
        cryptobert_content_score=cryptobert_content_score,
        cryptobert_title_and_description_sentiment=cryptobert_title_and_description_sentiment,
        cryptobert_title_and_description_score=cryptobert_title_and_description_score,
        finbert_content_sentiment=finbert_content_sentiment,
        finbert_content_score=finbert_content_score,
        finbert_title_and_description_sentiment=finbert_title_and_description_sentiment,
        finbert_title_and_description_score=finbert_title_and_description_score,
        llama_3_8b_content_sentiment=llama_3_8b_content_sentiment,
        llama_3_8b_content_score=llama_3_8b_content_score,
        llama_3_8b_title_and_description_sentiment=llama_3_8b_title_and_description_sentiment,
        llama_3_8b_title_and_description_score=llama_3_8b_title_and_description_score,
        phi_3_5_mini_content_sentiment=phi_3_5_mini_content_sentiment,
        phi_3_5_mini_content_score=phi_3_5_mini_content_score,
        phi_3_5_mini_title_and_description_sentiment=phi_3_5_mini_title_and_description_sentiment,
        phi_3_5_mini_title_and_description_score=phi_3_5_mini_title_and_description_score,
        mistral_7b_v0_3_content_sentiment=mistral_7b_v0_3_content_sentiment,
        mistral_7b_v0_3_content_score=mistral_7b_v0_3_content_score,
        mistral_7b_v0_3_title_and_description_sentiment=mistral_7b_v0_3_title_and_description_sentiment,
        mistral_7b_v0_3_title_and_description_score=mistral_7b_v0_3_title_and_description_score,
    )

    return result
