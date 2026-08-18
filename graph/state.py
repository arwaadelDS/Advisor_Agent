# graph/state.py
from typing import TypedDict, Optional
from schemas import SQLQueryResult, RAGSearchResult

class AdvisorState(TypedDict):
    messages: list         
    client_id: str         
    session_id: str
    next_agent: Optional[str]      
    sql_result: Optional[SQLQueryResult]
    rag_context: Optional[RAGSearchResult]
    search_result: Optional[str]