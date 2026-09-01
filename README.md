# HackBattle-ChatBot
RAG chatbot answering FAQs about IEEE-CS VIT HackBattle, grounded strictly in the official event document. Hybrid retrieval (bge-large embeddings + BM25, fused with RRF) over ChromaDB, generation via Groq, served as a FastAPI endpoint on Modal. Refuses off-topic questions instead of guessing.
