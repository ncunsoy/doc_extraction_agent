# Introduction to RAG Systems
RAG (Retrieval-Augmented Generation) is an architecture that allows large language models (LLMs) to access external information sources. This helps prevent model hallucinations and enables generating responses based on enterprise data.

## 1. Vector Databases (Vector DB)
Vector databases store numerical vectors (embeddings) that represent the semantic equivalents of words and sentences. Traditional databases look for keyword matches, while vector databases consider semantic proximity. For example, the words "köpek" and "enik" may be written differently but are very close in vector space.

## 2. Text Chunking Strategies
Chunking is the process of splitting large texts into smaller pieces that fit within the AI's context window. Pieces that are too large dilute context and confuse the AI. Pieces that are too small break holistic meaning. Therefore, recursive chunking strategies provided by tools like LangChain are critical.
