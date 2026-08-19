import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import pandas as pd

class DataManager:
    def __init__(self, data_path):
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection_name = "stores"
        self.dataset = pd.read_csv(data_path)

        EMBEDDING_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name= EMBEDDING_MODEL
        )

        self.raw_data = []
        
        self.chunks = []
        self.metadatas = []
        self.collection = None

        self.bm25_docs = []
        self.bm25_tokens = []
        self.bm25 = None

        self.reranker = CrossEncoder("BAAI/bge-reranker-base")  
    
    def _rerank(self, query, docs):
        reranker = self.reranker

        pairs = [(query, doc) for doc in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked]

# -------------------[ Chroma ]----------------------------------
    def _load_chroma_data(self):
        self.chunks = []
        self.metadatas = []

        print("start chunking...")
        for row in self.dataset:
            self.chunks.append(row['Story'])
            self.metadatas.append({
                ...
            })

    def _create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_func
        )
        print(f"Collection '{self.collection_name}' is ready.")
        
    def _add_documents(self, batch_size=64):
        if not self.chunks:
            return

        for i in range(0, len(self.chunks), batch_size):
            batch_chunks = self.chunks[i:i+batch_size]
            batch_metadatas = self.metadatas[i:i+batch_size]
            batch_ids = [str(x) for x in range(i, i + len(batch_chunks))]

            self.collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        print("Documents added to vector DB successfully.")

    def _chroma_search(self, query, n_results=1):
        if self.collection is None:
            self._create_collection()
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "distances"]
        )

        return results["documents"][0]

# -------------------[ BM25 ]----------------------------------
    def _normalize_arabic(self, text):
        # Remove diacritics (harakat)
        arabic_diacritics = re.compile(r"[\u064B-\u0652]")
        text = re.sub(arabic_diacritics, "", text)
        
        # Normalize letters
        text = re.sub(r"[أإآ]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        
        return text.strip()
    
    def _load_bm25_data(self):
        self.bm25_docs = []

        for row in self.raw_data:
            bm25_doc = f"""البيت: {row['verse']}\n المفردات: {row['vocabulary']}\n المعنى: {row['meaning']}\n الاعراب: {row['grammar']}"""
            self.bm25_docs.append(bm25_doc)

        self.bm25_tokens = [self._normalize_arabic(doc).split() for doc in self.bm25_docs]

    def _build_bm25(self):
        self.bm25 = BM25Okapi(self.bm25_tokens)
        print("BM25 index built.")

    def _preprocess_query(self, query):
        STOP_WORDS = {
            "في", "من", "على", "عن", "إلى", "مع", "قد", "لقد",
            "كان", "يكون", "هو", "هي", "هذا", "هذه", "ذلك", "تلك",
            "ثم", "إذا", "بين", "عند", "له", "لها", "إليه", "عليه"
        }

        query_tokens = self._normalize_arabic(query).split()

        return [t for t in query_tokens if t not in STOP_WORDS]

    def _bm25_search(self, query, n_results=3):
        query_tokens = self._preprocess_query(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        return [self.bm25_docs[i] for i in top_indices]

#---------------------------------------------------------------------------------------------------------------------------------------------------------
    def prepare(self):
        print("creat or get collection")
        self._create_collection()
        self._load_dataset()

        if not self.raw_data:
            raise ValueError("Dataset is empty")

        self._load_bm25_data()
        self._build_bm25()

        if self.collection.count() == 0:
            print("Empty collection → Building DB")
            self._load_chroma_data()
            self._add_documents()
        else:
            print("Existing collection loaded.")

    def search(self, query, n_results=10):
        vector_results = self._chroma_search(query, n_results)
        bm25_results = self._bm25_search(query, n_results)

        candidates = list(dict.fromkeys(vector_results + bm25_results))
        ranked = self._rerank(query, candidates)

        return ranked[:n_results]