import operator
from typing import Annotated, List, Sequence, TypedDict
from uuid import uuid4

from langchain_core.language_models import LLM
from langchain_core.language_models.base import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import chain
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.graph.state import StateGraph

# Define reusable prompt templates
class Prompts:
    # Prompt for generating a search query from conversation history
    search_prompt = PromptTemplate.from_template(
        """Given the conversation below, come up with a search query to look up.

This search query can be either a few words or question.

Return ONLY this search query, nothing more.

>>> Conversation:
{conversation}
>>> END OF CONVERSATION

Remember, return ONLY the search query that will help you when formulating a response to the above conversation."""
    )

    # Prompt for generating a response based on context
    response_prompt_template = """{instructions}

Respond to the user using ONLY the context provided below. Do not make anything up.

{context}"""

class InputPreprocessor:
    def __init__(self, llm: LLM):
        self.llm = LLM(llm)
    async def preprocess(self, user_query: str) -> str:
        prompt = f"""
               Given the user's question below, summarize it into a concise query for legal retrieval:

               User Question: {user_query}

               Simplified Query:"""

        # using LLM converting
        simplified_query = await self.llm.ainvoke(prompt)
        return simplified_query.strip()


# Define agent state structure
class AgentState(TypedDict):
    # List of conversation messages
    messages: Annotated[List[BaseMessage], add_messages_liberal]
    # Message count, with incremental updates
    msg_count: Annotated[int, operator.add]

# Class to handle search query generation
class SearchQueryGenerator:
    def __init__(self, llm: LanguageModelLike):
        self.llm = llm

    @chain
    async def generate(self, messages: Sequence[BaseMessage]) -> str:
        """Generate a search query from conversation messages."""
        convo = []
        for m in messages:
            # Process AI messages without function calls
            if isinstance(m, AIMessage):
                if "function_call" not in m.additional_kwargs:
                    convo.append(f"AI: {m.content}")
            # Process human messages
            if isinstance(m, HumanMessage):
                convo.append(f"Human: {m.content}")
        # Combine conversation into a single string
        conversation = "\n".join(convo)
        # Use prompt to generate a search query
        prompt = await Prompts.search_prompt.ainvoke({"conversation": conversation})
        response = await self.llm.ainvoke(prompt, {"tags": ["nostream"]})
        return response

# Class to manage retrieval operations
class RetrievalManager:
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    async def retrieve(self, query: str) -> str:
        """Retrieve content based on the given query."""
        return await self.retriever.ainvoke(query)

# Class to handle message processing for generating responses
class MessageProcessor:
    def __init__(self, llm: LanguageModelLike, system_message: str):
        self.llm = llm
        self.system_message = system_message

    def prepare_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Prepare messages by combining context and system instructions."""
        chat_history = []
        for m in messages:
            # Include AI messages without tool calls
            if isinstance(m, AIMessage):
                if not m.tool_calls:
                    chat_history.append(m)
            # Include human messages
            if isinstance(m, HumanMessage):
                chat_history.append(m)

        # Extract and format the context
        response = messages[-1].content
        content = "\n".join([d.page_content for d in response])
        return [
            SystemMessage(
                content=Prompts.response_prompt_template.format(
                    instructions=self.system_message, context=content
                )
            )
        ] + chat_history

    def generate_response(self, messages: List[BaseMessage]) -> AIMessage:
        """Generate a response from the LLM based on prepared messages."""
        return self.llm.invoke(self.prepare_messages(messages))

# Class to orchestrate the workflow
class RetrievalExecutor:
    def __init__(
        self,
        llm: LanguageModelLike,
        retriever: BaseRetriever,
        system_message: str,
        checkpoint: BaseCheckpointSaver,
    ):
        self.llm = llm
        self.retriever = retriever
        self.system_message = system_message
        self.checkpoint = checkpoint

        # Initialize components
        self.query_generator = SearchQueryGenerator(llm)
        self.retrieval_manager = RetrievalManager(retriever)
        self.message_processor = MessageProcessor(llm, system_message)

    async def invoke_retrieval(self, state: AgentState) -> dict:
        """Generate a search query or use human input for retrieval."""
        messages = state["messages"]
        if len(messages) == 1:
            # If there is only one message, use it as the query
            human_input = messages[-1]["content"]
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": uuid4().hex,
                                "name": "retrieval",
                                "args": {"query": human_input},
                            }
                        ],
                    )
                ]
            }
        else:
            # Generate a search query from conversation history
            search_query = await self.query_generator.generate(messages)
            return {
                "messages": [
                    AIMessage(
                        id=search_query.id,
                        content="",
                        tool_calls=[
                            {
                                "id": uuid4().hex,
                                "name": "retrieval",
                                "args": {"query": search_query.content},
                            }
                        ],
                    )
                ]
            }

    async def retrieve(self, state: AgentState) -> dict:
        """Fetch relevant content using the generated search query."""
        messages = state["messages"]
        params = messages[-1].tool_calls[0]
        query = params["args"]["query"]
        response = await self.retrieval_manager.retrieve(query)
        # Create a retrieval message with the result
        msg = LiberalToolMessage(
            name="retrieval", content=response, tool_call_id=params["id"]
        )
        return {"messages": [msg], "msg_count": 1}

    def call_model(self, state: AgentState) -> dict:
        """Generate a response from the LLM using retrieved content."""
        messages = state["messages"]
        response = self.message_processor.generate_response(messages)
        return {"messages": [response], "msg_count": 1}

    def get_workflow(self):
        """Define and compile the workflow for retrieval and response generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node("invoke_retrieval", self.invoke_retrieval)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("response", self.call_model)
        workflow.set_entry_point("invoke_retrieval")
        workflow.add_edge("invoke_retrieval", "retrieve")
        workflow.add_edge("retrieve", "response")
        workflow.add_edge("response", END)
        return workflow.compile(checkpointer=self.checkpoint)
