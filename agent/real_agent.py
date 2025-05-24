import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain.llms import HuggingFacePipeline
import torch
import transformers
from edit_model import edit_image
from gen_model import gen_image

load_dotenv()

tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false")  
project = os.environ.get("LANGCHAIN_PROJECT", "Default Project")
api_key = os.environ.get("LANGCHAIN_API_KEY")

pipe = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-Guard-2-8B",
    device=0 
)
llama3 = HuggingFacePipeline(pipeline=pipe)

# Define local LLM
#local_llm = 'llama3'
#llama3 = ChatOllama(model=local_llm, temperature=0)


# Graph state structure
class GraphState(TypedDict):
    prompt: str
    context: str
    prev_context: str
    prev_image: str
    query: str
    task_type: str
    questions: list[str]


def query_process(state: GraphState) -> dict:
    query = state["query"]
    prev_context = state["context"]

    cot_prompt = PromptTemplate.from_template("""
    You are a helpful assistant determining whether a user's query continues from a previous context or starts a new one.

    Previous context:
    {context}

    User query:
    "{query}"

    Step 1: Decide if the query continues from the previous context or starts a new one.
    - A query is a continuation if it:
    - Refers to or modifies content already mentioned (e.g., "make it brighter", "add more trees", "change the lighting").
    - Uses pronouns ("it", "that", "this") or context-dependent phrases ("make it", "now add") where their meaning can reasonably be inferred from the previous context.

    - A query is a new context if it:
    - Introduces unrelated elements (e.g., "create a medieval battle scene" after a beach scene).
    - Mentions new subjects not referred to in the previous context, and does not rely on existing concepts.
    
    Step 2: Only generate clarifying questions if the query is clearly vague **and** cannot be reasonably interpreted from context.

    Respond only in strict JSON format:

    If it is a continuation and clear:
    {{
    "is_new_context": false,
    "questions": []
    }}

    If it is a continuation but needs clarification:
    {{
    "is_new_context": false,
    "questions": [
        "clarifying question 1",
        "clarifying question 2"
    ]
    }}

    If it starts a new context:
    {{
    "is_new_context": true
    }}
    """)


    parser = JsonOutputParser()
    cot_chain = cot_prompt | llama3 | parser
    result = cot_chain.invoke({
        "query": query,
        "context": prev_context or "None"
    })

    # Log for debug
    print("Query:", query)
    print("Context:", prev_context)
    print("LLM result:", result)

    if result.get("is_new_context", False):
        return {
            **state,
            "prev_context": f"\nUser: {query}",  # clear previous context
            "task_type": "genprompt",
            "prompt": state.get("prompt", "")
        }

    # Continue previous context (append query and questions if any)
    continued_context = prev_context + f"\nUser: {query}"
    if result.get("questions"):
        for i, q in enumerate(result["questions"], 1):
            continued_context += f"\nAssistant follow-up {i}: {q}"

    return {
        **state,
        "prev_context": continued_context,
        "task_type": "solve_confusion" if result["questions"] else "genprompt",
        "questions": result.get("questions", []),
        "prompt": state.get("prompt", "")
    }


# Function: Prompt generation
def prompt_gen(state: GraphState) -> GraphState:
    context = state["prev_context"]
    query = state["query"]
    
    prompt_template = PromptTemplate.from_template("""
    You are an assistant helping prepare image generation tasks.

    Given the user query: "{query}", and previous context: "{context}", 
                                                   
    Respond *only* with a JSON object like this (no extra text, no explanations)::
    {{
    "task_type": "edit" or "generate",
    "prompt": "text prompt to generate or edit image"
    }}
    """)

    parser = JsonOutputParser()
    input_text = prompt_template.format(query=query, context=context)
    
    result = llama3.invoke(input_text)
    parsed = parser.invoke(result.content)
    print(parsed['prompt'])
    return {
        **state,
        "task_type": parsed["task_type"],
        "prompt": parsed["prompt"]
    }


# Function: Image editing
def editing_image(state: GraphState) -> GraphState:
    prev_image_url = state["prev_image"]
    edit_prompt = state["prompt"]

    new_image_url = edit_image(prev_image_url, edit_prompt)
    #new_image_url = 'it worked'
    print(f"Editing image {prev_image_url} with: {edit_prompt}")
    return {
        **state,
        "prev_image": new_image_url
    }

# Function: Image generation
def generate_image(state: GraphState) -> GraphState:
    gen_prompt = state["prompt"]

    new_image_url = gen_image(gen_prompt)
    #new_image_url = 'it worked'
    print(f"Generating image with: {gen_prompt}")
    return {
        **state,
        "prev_image": new_image_url
    }

def route_image(state: GraphState) -> str:
    return "editimage" if state["task_type"] == "edit" else "genimage"


# Create the graph
workflow = StateGraph(GraphState)

workflow.add_node("queryprocess", query_process)
workflow.add_node("genprompt", prompt_gen)
workflow.add_node("editimage", editing_image)
workflow.add_node("genimage", generate_image)
workflow.add_node("route_image", lambda state: state) 

# Routing from entry point
workflow.set_entry_point("queryprocess")
workflow.add_edge("queryprocess", "genprompt")
workflow.add_edge("genprompt", "route_image")

workflow.add_conditional_edges("route_image", route_image, {
    "editimage": "editimage",
    "genimage": "genimage"
})

workflow.add_edge("editimage", END)
workflow.add_edge("genimage", END)

# Compile
app = workflow.compile()


import gradio as gr

# Initial global state (could later be per-session for chat-style apps)
global_state = {
    "prompt": "",
    "context": "",
    "prev_context": "",
    "prev_image": "",  # Could point to a placeholder or be empty initially
    "query": "",
    "task_type": "",
    "questions": []
}

def process_query_full(user_query):
    state = global_state.copy()
    state["query"] = user_query
    result_state = app.invoke(state)
    global_state.update(result_state)
    
    img_url = result_state["prev_image"]
    context_log = result_state["prompt"]
    return img_url, context_log


demo = gr.Interface(
    fn=process_query_full,
    inputs=gr.Textbox(label="Enter your image request or edit command"),
    outputs=[
        gr.Image(type="filepath", label="Image"),
        gr.Textbox(label="Context History", lines=10)
    ],
    title="Smart Image Assistant",
    description="Uses LangGraph to understand and track image instructions."
)

demo.launch()

