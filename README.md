# ScholarBot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/FAISS-0055A4?style=for-the-badge&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
</p>


🔗 **Access the Streamlit Dashboard Here**: [ScholarBot on Streamlit](https://scholarbot.streamlit.app/)

ScholarBot is a question-answering chatbot that transforms any article into a conversational knowledge base. Just enter a webpage URL, and ScholarBot will let you ask deep, meaningful questions based on the article’s content in real time.

![ScholarBot Screenshot](ScholarBot%20Preview.png)
---

## 🧠 Project Overview

ScholarBot allows users to input a URL, automatically fetches the text from that page, and builds a vectorstore using FAISS and HuggingFace embeddings. It then uses LangChain’s RetrievalQA to query the data.

The result is a clean, focused, academic tool for summarizing, extracting, or deeply understanding content from any webpage, especially useful for research, note-taking, or exam prep.

---

## 🛠 Tech Stack

| Tool              | Purpose                                         |
|-------------------|-------------------------------------------------|
| Python            | Core programming language                       |
| Streamlit         | Frontend dashboard                              |
| LangChain         | Orchestrates retrieval-based QA pipeline        |
| HuggingFace       | Embedding model for semantic understanding      |
| FAISS             | Efficient vector similarity search              |
| Docker            | Containerization for easy deployment            |

---

## 💡 Key Features  

Accepts **any article URL** for analysis  
Provides **accurate, context-aware answers**  
Uses semantic search with HuggingFace embeddings  
Conversational UI built in Streamlit  
Modular backend ready for deployment 

---

👩‍🎓 Made with ♥ for scholars, researchers, and the curious mind.
