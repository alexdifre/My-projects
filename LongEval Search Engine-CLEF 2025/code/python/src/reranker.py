import torch
try:
    import torch_directml
    DML_AVAILABLE = True
except ImportError:
    DML_AVAILABLE = False
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2", device=None):
        if device:
            self.device = device
        elif DML_AVAILABLE:
            self.device = torch_directml.device()
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

    def rerank(self, query, documents, top_n=None):
        pairs = [(query, doc) for doc in documents]
        inputs = self.tokenizer.batch_encode_plus(
            pairs, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1).cpu()
            scores = torch.sigmoid(logits).numpy()*100
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        if top_n:
            ranked = ranked[:top_n]
        return [{"document": doc, "score": float(score)} for doc, score in ranked]

# FastAPI app
app = FastAPI()
reranker = CrossEncoderReranker()

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = None

class RerankResult(BaseModel):
    index: int
    relevance_score: float

class RerankResponse(BaseModel):
    results: List[RerankResult]

@app.post("/rerank", response_model=RerankResponse)
async def rerank_endpoint(request: RerankRequest):
    scores = reranker.rerank(request.query, request.documents, request.top_n)
    doc_to_index = {doc: i for i, doc in enumerate(request.documents)}
    indexed_scores = [
        {"index": doc_to_index[item["document"]], "relevance_score": item["score"]}
        for item in scores
    ]
    response = { "results": indexed_scores }
    return response

# Esempio d'uso:
if __name__ == "__main__":
    query = "1ere guerre mondiale"
    docs = [
        "Les causes principales de la Première Guerre mondiale et le rôle des alliances européennes.",
        "La vie des soldats dans les tranchées pendant la Première Guerre mondiale.",
        "Hello, how are you?",
    ]
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, docs)
    for item in results:
        print(f"Score: {item['score']:.4f} | Doc: {item['document']}")