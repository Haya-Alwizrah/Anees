import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq

class StoryGenerator:
    def __init__(self, data_path:str, api_key:str, model_name):
        self.client = chromadb.PersistentClient(path="datasets\\chroma_db")
        self.collection_name = "stories"
        
        df = pd.read_csv(data_path)
        self.dataset = df["Story"].dropna().astype(str).tolist()

        self.EMBEDDING_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
        self.embedding_func = None
        self.reranker = None
        
        self.chunks = []
        self.metadatas = []
        self.collection = None

        self.bm25_docs = []
        self.bm25_tokens = []
        self.bm25 = None

        self.is_prepared = False

        self.topic_queries = {
            "الصداقة": "قصة عن الصداقة والتعاون ومساعدة الأصدقاء",
            "الصدق": "قصة عن الصدق والأمانة وقول الحقيقة",
            "التعاون": "قصة عن التعاون والعمل الجماعي ومساعدة الآخرين",
            "احترام الوالدين": "قصة عن بر الوالدين واحترامهما وطاعتهما",
            "مساعدة الآخرين": "قصة عن مساعدة الآخرين والتعاطف معهم وتقديم الدعم",
            "المسؤولية": "قصة عن تحمل المسؤولية والاعتماد على النفس والقيام بالواجبات",
            "التسامح": "قصة عن التسامح والعفو واحترام الآخرين",
            "النظافة": "قصة عن النظافة الشخصية ونظافة البيئة والمحافظة على المكان"
        }
        self.llm = ChatGroq(model=model_name, temperature=0.7, api_key= api_key)
        
# -------------------[ Chroma ]----------------------------------
    def _load_chroma_data(self):
        self.chunks = []
        self.metadatas = []

        print("Start loading stories...")
        for i, row in enumerate(self.dataset):
            self.chunks.append(row)
            self.metadatas.append({
                "row_id": i
            })

        print(f"Loaded {len(self.chunks)} stories.")

    def _create_collection(self):
        if self.embedding_func is None:
            print("Loading embedding_func...")
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name= self.EMBEDDING_MODEL)
            print("embedding_func loaded.")

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
        self.bm25_docs =self.dataset.copy()
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

# -------------------[ Re: Ranker ]----------------------------------
    def _rerank(self, query, docs):
        if self.reranker is None:
            print("Loading reranker...")
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder("BAAI/bge-reranker-base")
            print("Reranker loaded.")

        reranker = self.reranker

        pairs = [(query, doc) for doc in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked]

#---------------------------------------------------------------------------------------------------------------------------------------------------------
    def prepare(self):
        print("creat or get collection")
        self._create_collection()

        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
    
        self._load_bm25_data()
        self._build_bm25()

        if self.collection.count() == 0:
            print("Empty collection → Building DB")
            self._load_chroma_data()
            self._add_documents()
        else:
            print("Existing collection loaded.")

        self.is_prepared = True

    def search(self, query, n_results=10):
        vector_results = self._chroma_search(query, n_results)
        bm25_results = self._bm25_search(query, n_results)

        candidates = list(dict.fromkeys(vector_results + bm25_results))
        ranked = self._rerank(query, candidates)

        return ranked[:n_results]


    def search(self, query, search_type=5, n_results=10, ):
        if search_type == 1: # only vector
            return self._chroma_search(query, n_results)

        elif search_type == 2: # onlu bm25
            return self._bm25_search(query, n_results)

        elif search_type == 3: # vector + bm25
            vector_results = self._chroma_search(query, n_results)
            bm25_results = self._bm25_search(query, n_results)

            candidates = list(dict.fromkeys(vector_results + bm25_results))
            return candidates[:n_results]

        elif search_type == 4: # vector + reranker
            vector_results = self._chroma_search(query, n_results)
            return self._rerank(query, vector_results)[:n_results]
        
        elif search_type == 5: # vector + bm25 + reranker
            vector_results = self._chroma_search(query, n_results)
            bm25_results = self._bm25_search(query, n_results)

            candidates = list(dict.fromkeys(vector_results + bm25_results))
            ranked = self._rerank(query, candidates)

            return ranked[:n_results]
        
        else:
            raise ValueError(f"Unknown search_type: {search_type}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------
    def generate_story(self, search_t:int=1, topic="الصدق", character="أرنب", n_results=5):
        if not self.is_prepared:
            self.prepare()
            
        if topic not in self.topic_queries:
            raise ValueError(f"Unknown topic: {topic}")

        query = self.topic_queries[topic]

        stories = self.search(query, search_t, n_results)
        context = "\n\n".join([f"القصة {i+1}:\n{story}" for i, story in enumerate(stories)])

        prompt = f"""
أنت كاتب قصص تعليمية للأطفال.

الموضوع المطلوب:
{topic}

وصف الموضوع:
{query}

استخدم القصص المرجعية التالية للاستفادة من أسلوبها وأفكارها:
{context}

اجعل بطل الذي تتحدث عنه القصة عبارة عن : {character}

اكتب قصة عربية جديدة ومناسبة للأطفال حول الموضوع المطلوب.

الشروط:
- لا تنسخ القصص المرجعية حرفيًا.
- أنشئ قصة جديدة ومبتكرة.
- استخدم لغة عربية واضحة وبسيطة.
- اجعل القصة مناسبة للأطفال.
- اجعل القصة تتضمن رسالة أو قيمة تعليمية مرتبطة بالموضوع.
- اجعل القصة 10 جمل فقط وبفقرة واحدة او فقرتين
"""

        response = self.llm.invoke(prompt)

        return response.content


    def generate_QA(self, story):
        prompt = f"""
أنت كاتب أسئلة تعليمية للأطفال استناداً على قصة.

القصة:
{story}

اكتب 5 اسئلة اختيار من متعدد بناء على القصة المعطاة
الشروط:
- اجعل اجابة واحدة صحيحة يمكن للطفل استنتاجها من القصة
- انشئ 3 اجوبه خاطئة احدها قريب من الاجابه الصحيحة 
- استخدم لغة عربية واضحة وبسيطة.

قم بارجاع الناتج بصيغة JSON التالية:
{{
    "questions": [
        {{
            "id": 1,
            "question": "اكتب السؤال هنا",
            "c_answer": "اكتب الإجابة الصحيحة هنا",
            "w_answer": [
                "الإجابة الخاطئة 1",
                "الإجابة الخاطئة 2",
                "الإجابة الخاطئة 3"
            ]
        }}
    ]
}}
"""

        response = self.llm.invoke(prompt)

        return response.content