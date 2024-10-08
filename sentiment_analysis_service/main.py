from fastapi import FastAPI
import logging
from dotenv import load_dotenv

from api.make_analysis import router as make_analysisrouter

load_dotenv()


app = FastAPI(title="Sentiment analysis for crypto news")
app.include_router(make_analysisrouter, prefix="/api", tags=["ml"])


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    # При запуске через отладчик PyCharm (и др. IDE) или через консоль файла main.py
    logging.info("Запуск backend компонента произведён через отладчик")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # При запуске через команду Uvicorn (пример: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000)
    logging.info("Запуск backend компонента произведён через команду python -m uvicorn")