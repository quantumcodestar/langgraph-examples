from typing import Annotated
import uuid
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from PIL import Image
import io
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.messages import AIMessage, HumanMessage

# Represents your agent global state
class State(TypedDict):
    email_subject: str
    email_body: str
    messages: Annotated[list, add_messages]

# StateGraph is your actual agent
graph_builder = StateGraph(State)

# Connect to your local Ollama endpoint by modifying the base URL.
from langchain_openai import ChatOpenAI
model_id = "llama3.2:latest"
base_url = "http://localhost:11434/v1"

llm = ChatOpenAI(model= model_id, base_url= base_url, temperature=0, max_tokens=4096, api_key="Not required with Ollama")

def writer(state: State):
    response = llm.invoke(state["messages"])
    return {"email_body": response.content, "messages": [response]}

def subject(state: State):
    email_body = state["email_body"]
    prompt = f"Write an email subject based on this email body: {email_body}. Exclude any body, or comments; just provide the email subject."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"email_subject": response.content, "messages": [response]}

graph_builder.add_node("writer", writer)
graph_builder.add_node("subject", subject)
graph_builder.add_edge("writer", 'subject')
graph_builder.set_entry_point("writer")
graph_builder.set_finish_point("subject")

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

request = """Generate only the body of a polite email to Joan Smith from AmazingInternet, 
requesting a solution to an ongoing issue with my internet connection. 

The connection has been failing every day for at least 3 hours since 01/01/2025. 
Exclude any subject, title, or comments; just provide the email body."""

thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

response = ""
response = graph.invoke(
    {"messages": [("user", request)]}, config, stream_mode="values"
)

# Print agent state snapshot
snapshot = graph.get_state(config)
print(snapshot.values)

# Optional: Save your graph diagram
image_bytes = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )

image = Image.open(io.BytesIO(image_bytes))
image.save('output_graph.png') 
