import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import HuggingFacePipeline
import torch
import transformers
import requests
import gradio as gr

BASE_URL =  "Run fast_api.ipynb for url to fill"

load_dotenv()

tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false")  
project = os.environ.get("LANGCHAIN_PROJECT", "Default Project")
api_key = os.environ.get("LANGCHAIN_API_KEY")

local_llm = 'deepseek-coder:6.7b'

llama3 = ChatOllama(model=local_llm, temperature=0)


class GraphState(TypedDict):
    prompt: str
    prev_context: str
    prev_image: str
    query: str
    task_type: str
    questions: list[str]


def query_process(state: GraphState) -> dict:
    query = state["query"]
    prev_context = state["prev_context"] or "None"
    
    # Agent 1: Context Classifier
    context_classifier_prompt = PromptTemplate.from_template("""
You are a context classification expert. Determine if the query continues previous context or starts new context.

### Previous Context:
{prev_context}

### Current Query:
{query}

---
### Reasoning Steps:
1. **Evaluate Questioning Nature**: 
   - If query contains evaluation/questioning words ("why", "how", "explain", "analyze", "evaluate") **or**  the query is answering to a question of a assistant
   - Then conclude: CONTINUATION → Proceed to output

2. **Reference Analysis**:
   - If query **contains**: 
     a) Pronouns ("it", "that", "this")
     b) **Continuation phrases** ("now add", "then make", "adjust the","add" , "make", "remove", "replace", "change")
   - Then conclude: CONTINUATION → Proceed to output

3. **New Subject Check** (ONLY if Steps 1-2 are inapplicable):
   - Does query introduce completely new subjects/scenarios **without any continuation phrases** ("now add", "then make", "adjust the","add" , "make", "remove", "replace", "change")
   - Is there zero relationship to previous context?
   - If YES → NEW CONTEXT
   - If NO → CONTINUATION

**Output Format (strict JSON):**
{{
  "is_new_context": boolean,
  "reasoning": "brief explanation"
}}
---
Important instructions:
- Output must be strictly valid JSON.
- Do not include any natural language explanation.
- Do not use markdown, headers, or preambles.
- Think step by step, but only output JSON.
""")

    context_chain = context_classifier_prompt | llama3 | JsonOutputParser()
    context_result = context_chain.invoke({"query": query, "prev_context": prev_context})
    print(context_result)
    # Agent 2: Ambiguity Detector
    ambiguity_prompt = PromptTemplate.from_template("""
You are an ambiguity detection expert. Jointly analyze the query and previous context to identify critical ambiguities.

### Context Type: {context_type}
### Previous Context: {prev_context}
### Current Query: {query}

---
### Unified Analysis Approach:
1. **Cross-Examine Context and Query**:
   - Identify elements from previous context referenced in query
   - Verify if references are unambiguous and consistent
   - Detect contradictions between context and query
   - Check if vague terms are resolvable through context

2. **Resolved References Verification**:
   - For pronouns ("it", "that", "this"):
     a) Match to explicit elements in previous context
     b) Confirm single unambiguous referent exists
     c) If multiple possible referents → Flag ambiguity

3. **Consistency Validation**:
   - Physical/logical consistency (day/night, spatial relationships)
   - Stylistic consistency (art style, palette)
   - Thematic consistency (mood, genre)

4. **Essential Completeness Check**:
   - For NEW CONTEXT: Verify query contains subject + setting + key details
   - For CONTINUATION: Verify modifications are actionable

---
### Critical Ambiguity Detection Rules:
1. **####NEVER#### ask about editing purpose** - assume modification intent
2. **Flag ONLY when**:
   a) Unresolvable reference conflict exists
   b) Physical/logical contradiction can't be reconciled
   c) Vague term lacks context-based interpretation
   d) Essential element remains missing after context+query analysis

3. **Never flag**:
   - References with clear context antecedents
   - Vague terms with reasonable defaults
   - Stylistic requests without specifics
   - Already resolved ambiguities in conversation history

4. **Question Threshold**:
   - Generate MAX 1 question only if processing is impossible
   - Question must address a specific show-stopping ambiguity

---                                                 

**Output Format (strict JSON):**
{{
  "questions": ["single specific question"]  // Empty array if no critical ambiguity
}}

---
Important instructions:
- Output STRICT JSON ONLY (no additional text)
- Analyze context and query as a unified whole
- Consider entire conversation history
- Only flag ambiguities that prevent task execution
- Do not include any natural language explanation.
- Do not use markdown, headers, or preambles.                                       
""")
    context_type = "NEW CONTEXT" if context_result["is_new_context"] else "CONTINUATION"
    ambiguity_chain = ambiguity_prompt | llama3 | JsonOutputParser()
    ambiguity_result = ambiguity_chain.invoke({
        "query": query,
        "prev_context": prev_context,
        "context_type": context_type
    })
    print("LLM result:", ambiguity_result)
    if not context_result["is_new_context"] :
        continued_context = prev_context + f"\nUser: {query}"
        if ambiguity_result["questions"]:
            for i, q in enumerate(ambiguity_result["questions"], 1):
                continued_context += f"\nAssistant follow-up {i}: {q}"
    else :
        continued_context = 'None'
    return {
        **state,
        "prev_context": continued_context,
        "task_type": "solve_confusion" if ambiguity_result["questions"] else "genprompt",
        "questions": ambiguity_result["questions"],
    }



# Function: Prompt generation
def prompt_gen(state: GraphState) -> GraphState:
    context = state["prev_context"]
    query = state["query"]
    
    prompt_template = PromptTemplate.from_template("""
    You are an assistant helping prepare image generation or image editing tasks
                                                

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

def route_task_type(state):
    if state["task_type"] == "genprompt":
        return "genprompt"
    return "__end__"

# Function: Image editing
def editing_image(state: GraphState) -> GraphState:
    prev_image_url = state["prev_image"]
    edit_prompt = state["prompt"]

    response = requests.post(f"{BASE_URL}/edit", json={
        "url": prev_image_url,
        "prompt": edit_prompt
    })

    new_image_url = response.json()["url"]
    #new_image_url = 'it worked'
    print(f"Editing image {prev_image_url} with: {edit_prompt}")
    return {
        **state,
        "prev_image": new_image_url
    }

# Function: Image generation
def generate_image(state: GraphState) -> GraphState:
    gen_prompt = state["prompt"]
    response = requests.post(f"{BASE_URL}/generate", json={"prompt": gen_prompt})
    new_image_url = response.json()["url"]
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

# Routing from entry point
workflow.set_entry_point("queryprocess")
workflow.add_conditional_edges("queryprocess", route_task_type)
workflow.add_conditional_edges("genprompt", route_image, {
    "editimage": "editimage",
    "genimage": "genimage"
})

workflow.add_edge("editimage", END)
workflow.add_edge("genimage", END)

# Compile
app = workflow.compile()

# Initial global state (could later be per-session for chat-style apps)
global_state = {
    "prompt": "",
    "context": "",
    "prev_context": "",
    "prev_image": "",  
    "query": "",
    "task_type": "",
    "questions": [],
    "chat_history": []
}

from PIL import Image
from io import BytesIO
import requests

def process_query_full(user_query):
    state = global_state.copy()
    state["query"] = user_query
    result_state = app.invoke(state)
    global_state.update(result_state)

    img_url = result_state.get("prev_image", None)
    context_log = result_state.get("prev_context", "")

    # If task_type is genprompt and there's a valid URL, return the URL string
    if result_state.get("task_type") != "solve_confusion" and img_url:
        image = img_url  # Just pass the URL string
    else:
        image = None

    return image, context_log

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



