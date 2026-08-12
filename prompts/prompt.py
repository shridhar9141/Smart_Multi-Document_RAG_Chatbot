from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt to rephrase follow-up questions using chat history
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# System prompt for answering question based on retrieved context and history
SYSTEM_PROMPT = """You are a helpful AI Assistant.

Answer ONLY using the context below.

If the answer is not found in the context, reply:
"I couldn't find that information in the uploaded document."

Context:
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)