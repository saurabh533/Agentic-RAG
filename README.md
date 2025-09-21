uses a Groq-hosted LLM (via langchain-groq)

pulls a prompt template from LangChain Prompt Hub

calls tools including a Retriever backed by Supabase (Postgres + pgvector)

orchestrates reasoning + tool-use with LangChain (agents/tool-calling)
