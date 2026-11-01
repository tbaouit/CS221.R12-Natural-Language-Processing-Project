import numpy as np 
from rank_bm25 import BM250kapi 

def tokenize(text: str, scheme: str) -> list[str]: 
    text_clean = text.lower().strip()

    if scheme == "word":
        return text_clean.split()
    elif scheme == "bigram": 
        return [text_clean[i: i+2] for i in range(len(text_clean) - 1)]
    elif scheme == "trigram": 
        return [text_clean[i: i+3] for i in range(len(text_clean) - 2)]
    else: 
        raise ValueError(f"Không hỗ trợ kiểu tách từ: {scheme}")

class BM25Rertriever: 
    def __init__(self, train_data: list[dict], scheme: str="word"): 
        self.train_data = train_data
        self.scheme = scheme

        tokenized_corpus = [
            tokenize(item["Text"], scheme) for item in self.train_data
        ] 

    def retrieve(self, query_text: str, k: int, exclude_id: str = None) -> list[dict]: 
        return 