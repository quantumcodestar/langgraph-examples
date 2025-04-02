# Dependencies required for state annotation
from typing import Annotated
from typing_extensions import TypedDict

# LangGraph dependencies
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Required for image export
from PIL import Image
import io

# Other dependencies
import uuid

# Represents your agent global state
class State(TypedDict):
    email_subject: str
    email_body: str
    messages: Annotated[list, add_messages]

# StateGraph is your actual agent
graph_builder = StateGraph(State)

# Connect to your local Ollama endpoint by modifying the base URL.
model_id = "llama3.2:latest"
base_url = "http://localhost:11434/v1"
api_key = "Not required with local Ollama" # On production load from secrets manager, vault, secure storage, etc

llm = ChatOpenAI(model= model_id, base_url= base_url, temperature=0, max_tokens=4096, api_key=api_key)

def email_writer(state: State):
    response = llm.invoke(state["messages"])
    return {"email_body": response.content, "messages": [response]}

def subject_generator(state: State):
    email_body = state["email_body"]
    prompt = f"Write an email subject based on this email body: {email_body}. Exclude any body, or comments; just provide the email subject."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"email_subject": response.content, "messages": [response]}

graph_builder.add_node("email_writer", email_writer)
graph_builder.add_node("subject_generator", subject_generator)
graph_builder.add_edge("email_writer", 'subject_generator')
graph_builder.set_entry_point("email_writer")
graph_builder.set_finish_point("subject_generator")

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Optional: Save your graph diagram
image_bytes = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )

image = Image.open(io.BytesIO(image_bytes))
image.save('output_graph.png') 



thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

request = """Generate only the body of a polite email to Joan Smith from AmazingInternet, 
requesting a solution to an ongoing issue with my internet connection. 

The connection has been failing every day for at least 3 hours since 01/01/2025. 
Exclude any subject, title, or comments; just provide the email body."""

response = ""
response = graph.invoke(
    {"messages": [("user", request)]}, config, stream_mode="values"
)

# Print agent state snapshot
snapshot = graph.get_state(config)

# Feel free to use the values as needed, or you can create another node to automate the email sending

print("Subject:")
print(snapshot.values["email_subject"])

print("Body:")
print(snapshot.values["email_body"])
