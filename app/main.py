from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load the model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/predict/")
async def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiments = ["negative", "neutral", "positive"]
    pred = sentiments[torch.argmax(probs).item()]
    return {"sentiment": pred, "probabilities": probs.tolist()}