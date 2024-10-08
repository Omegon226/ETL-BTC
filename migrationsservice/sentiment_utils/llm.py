from huggingface_hub import InferenceClient
import signal
from functools import wraps
import re


LLM_PROMT = f"""Conduct a sentiment analysis of the following news story and provide a response in the following format:

- **Tonality**: Positive/Negative/Neutral
- **Confidence**: a number between 0 and 1

*Do not write anything other than these pauncts*

News text: """ # Здесб юудет даписываться текст новости

HUGGINGFACE_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-27b-it",
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
]


def timeout(seconds, fallback=None):
    """
    ВНИМАНИЕ! ЭТА ФУНКЦИЯ РАБОТАЕТ ТОЛЬКО НА UNIX СИСТЕМАХ
    """

    def decorator(func):
        def _handle_timeout(signum, frame):
            raise Exception(f"Функция {func.__name__} превысила лимит времени в {seconds} секунд")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Устанавливаем обработчик сигнала
            signal.signal(signal.SIGALRM, _handle_timeout)
            # Запускаем таймер
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except Exception as error:
                print("Откат по fallback")
                print(error)
                return fallback
            finally:
                # Отключаем сигнал
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def extract_data_from_llm_response(text):
    tonality_pattern = r'\*\*(Tonality|Sentiment)\*\*:\s*\/?([A-Za-z]+)'
    confidence_pattern = r'\*\*Confidence\*\*:\s*([0-9.]+)'

    tonality_match = re.search(tonality_pattern, text)
    confidence_match = re.search(confidence_pattern, text)

    tonality = tonality_match.group(2) if tonality_match else None
    confidence = float(confidence_match.group(1)) if confidence_match else None

    return {
        'label': tonality,
        'score': confidence
    }


@timeout(
    seconds=30,
    fallback=({'label': 'None', 'score': 0}, '')
)
def llm_huggingface_sentiment_calculate(client, news_text, model="mistralai/Mistral-7B-Instruct-v0.3", max_tokens=50,
                                        temperature=0.8):
    global HUGGINGFACE_MODELS, LLM_PROMT
    if model not in HUGGINGFACE_MODELS:
        raise Exception("Нет информации о предоставленной модели")

    try:
        content = LLM_PROMT + f'"{news_text}"'
        llm_promt_result = ""

        for message in client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature
        ):
            llm_promt_result += message.choices[0].delta.content

        if llm_promt_result == "":
            return {'label': 'None', 'score': 0}, llm_promt_result
        else:
            result = extract_data_from_llm_response(llm_promt_result)
            if result['score'] is None:
                result['score'] = 0

            if result['label'] not in ['Neutral', 'Positive', 'Negative']:
                return {'label': 'None', 'score': 0}, llm_promt_result
            else:
                return result, llm_promt_result
    except Exception as error:
        raise error

