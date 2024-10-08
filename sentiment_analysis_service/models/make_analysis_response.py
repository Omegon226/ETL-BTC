from pydantic import BaseModel


class MakeAnalysisResponse(BaseModel):
    embedding: list

    cryptobert_content_sentiment: str
    cryptobert_content_score: float
    cryptobert_title_and_description_sentiment: str
    cryptobert_title_and_description_score: float

    finbert_content_sentiment: str
    finbert_content_score: float
    finbert_title_and_description_sentiment: str
    finbert_title_and_description_score: float

    llama_3_8b_content_sentiment: str
    llama_3_8b_content_score: float
    llama_3_8b_title_and_description_sentiment: str
    llama_3_8b_title_and_description_score: float

    phi_3_5_mini_content_sentiment: str
    phi_3_5_mini_content_score: float
    phi_3_5_mini_title_and_description_sentiment: str
    phi_3_5_mini_title_and_description_score: float

    mistral_7b_v0_3_content_sentiment: str
    mistral_7b_v0_3_content_score: float
    mistral_7b_v0_3_title_and_description_sentiment: str
    mistral_7b_v0_3_title_and_description_score: float
