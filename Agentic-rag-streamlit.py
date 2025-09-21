# --- imports & setup ---
import os
from dotenv import load_dotenv

import streamlit as st

# LangChain core/agents/prompts
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub

# Groq LLM
from langchain_groq import ChatGroq

# Embeddings: BGE-M3 (FlagEmbedding)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Vector store: Supabase
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# Tools
from langchain_core.tools import tool

# -------------------------
# env
# -------------------------
load_dotenv()
# Expect: SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY in your .env
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# -------------------------
# embeddings (BGE-M3)
# -------------------------
# NOTE: Requires `pip install FlagEmbedding`
# BGE-M3 dense vector dimension is 1024. Your Supabase pgvector column must match.
bge_model_name = "BAAI/bge-m3"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=bge_model_name,
    query_instruction="Represent this sentence for searching relevant data:",
    model_kwargs={"device": "cpu"},          # use "cuda" if you have a GPU
    encode_kwargs={"normalize_embeddings": True},  # keep normalization here (avoids pydantic error)
)

# -------------------------
# vector store
# -------------------------
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",        # must have a 1024-d pgvector column if using BGE-M3
    query_name="match_documents",
)

# -------------------------
# LLM (Groq)
# -------------------------
# `pip install langchain-groq` and set GROQ_API_KEY
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
)

# -------------------------
# Prompt from LangChain Hub
# -------------------------
prompt = hub.pull("hwchase17/openai-functions-agent")

# -------------------------
# Tool: retriever
# -------------------------
from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve]

# -------------------------
# Agent & executor
# -------------------------
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# input box
user_question = st.chat_input("How are you?")

# handle submit
if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    # agent invoke
    result = agent_executor.invoke({"input": user_question, "chat_history": st.session_state.messages})
    ai_message = result["output"]

    with st.chat_message("assistant"):
        st.markdown(ai_message)
    st.session_state.messages.append(AIMessage(ai_message))