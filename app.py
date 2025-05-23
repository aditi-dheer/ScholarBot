# import langchain dependencies
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from langchain_community.vectorstores import FAISS
from pydantic import Field
import google.generativeai as genai

# streamlit
import streamlit as st

# for API calls
class GeminiLLM(LLM):
    api_key: str = Field(...)
    model: str = "gemini-2.0-flash"
    temperature: float = 0.5
    max_tokens: int = 200

    @property
    def _llm_type(self) -> str:
        return "google-gemini"
    
    # calls gemini and returns response
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        genai.configure(api_key=self.api_key)
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        # return response
        return response.text
    
# sample placeholder url about AI
website_url = st.text_input("Enter website URL:", "https://en.wikipedia.org/wiki/Artificial_intelligence")

@st.cache_resource
def load_website_data(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

if website_url:
    vectorstore = load_website_data(website_url)
    retriever = vectorstore.as_retriever()

    # Setup Gemini LLM

    gemini_llm = GeminiLLM(api_key=st.secrets["GEMINI"]["API_KEY"])

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=gemini_llm,
        chain_type="stuff",
        retriever=retriever
    )

# streamlit page with title and explanatory text
st.title("ScholarBot")

st.markdown("""
**Note:**  
This tool is for **academic purposes only** and is designed to provide **completely accurate information**.  
It does **not** accept typos or incorrect inputs because accuracy is critical.  

The default URL is a placeholder article about AI — you can replace it with **any article URL** you need information on.  

You can also use this tool to **summarize articles, extract detailed information or ask questions** from the content you provide.
""")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
prompt = st.chat_input("Enter Your Prompt Here")

# to display prompt history
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content':prompt})
    response = qa_chain.run(prompt)
    st.chat_message('assistant').markdown(response)

    st.session_state.messages.append(
        {'role': 'assistant', 'content': response}
    )