from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Define a request model for FastAPI
class TextRequest(BaseModel):
    text: str

# Load the model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/predict/")
async def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiments = ["negative", "neutral", "positive"]
    pred = sentiments[torch.argmax(probs).item()]
    return {"sentiment": pred, "probabilities": probs.tolist()}
