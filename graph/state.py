# graph/state.py
from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from schemas import SQLQueryResult, RAGSearchResult

class AdvisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]        
    client_id: str         
    session_id: str
    next_agent: Optional[str]      
    sql_result: Optional[SQLQueryResult]
    rag_context: Optional[RAGSearchResult]
    search_result: Optional[str]