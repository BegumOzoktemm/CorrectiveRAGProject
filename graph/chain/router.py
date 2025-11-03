from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal


class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or vectorstore"
    ) # üç nokta ikisinden birini seç anlamına getiriyor.

llm = ChatOpenAI(temperature=0)
structured_lm_router = llm.with_structured_output(RouteQuery) #llm e bu ikisinden birini dikkate al demek oluyor

system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents,prompt engineering and adversarial attacks.
Use the vectorstore for questions on these topic.For all else, use web-search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

question_router = route_prompt | structured_lm_router
