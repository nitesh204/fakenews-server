# ai_classifier.py
from transformers import pipeline

class FakeNewsClassifier:
    def __init__(self):
        print("Loading AI model... (this may take a few seconds)")
        self.classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        self.label_map = {"LABEL_0": "FAKE", "LABEL_1": "REAL"}

    def predict(self, text: str):
        if not text or text.strip() == "":
            return "UNKNOWN"
        result = self.classifier(text[:512])[0]  # Limit to 512 tokens
        return  self.label_map.get(result['label'], "UNKNOWN")  # returns "FAKE" or "REAL"
