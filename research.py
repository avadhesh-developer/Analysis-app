import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage,AIMessage,SystemMessage
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode
from langchain.tools import BaseTool
from typing import TypedDict, Annotated, List, Dict, Any
import operator

###Clip Model
import os
from dotenv import load_dotenv
load_dotenv()

## set up the environment
os.environ["OLLAMA_API_KEY"]=os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")

def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.0
    )

### initialize the Clip Model for unified embeddings
clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

#global storage for document and embeding
all_docs = []
all_embeddings = []
image_data_store = {}  # Store actual image data for LLM
table_data_store={}
vector_store=None

# State definition for the multi-agent system
class AgentState(TypedDict):
    messages: Annotated[List[Any],operator.add]
    query:str
    retrieved_docs:List[Document]
    text_analysis:str
    image_analysis: str
    table_analysis: str
    final_answer: str
    current_agent: str
    task_complete: bool


### Embedding functions
def embed_image(image_data):
    """Embed Image Using CLIP"""
    if isinstance(image_data,str):
        image=Image.open(image_data).convert("RGB")
    else:
        image=image_data

    input=clip_processor(images=image,return_tensors="pt")
    with torch.no_grad():
        features=clip_model.get_image_features(**input)
        # Normalize embeddings to unit vector
        features=features/features.norm(dim=1,keepdim=True)
        return features.squeeze().numpy()
    
def embed_text(text):
    """Embed Text Using CLIP"""
    input=clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features=clip_model.get_text_features(**input)
        # Normalize embeddings
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
def extract_tables_from_page(page):
    """Extract tables from PDF page"""
    tables = []
    try:
        # Find tables using PyMuPDF
        tabs = page.find_tables()
        for tab_index, tab in enumerate(tabs):
            # Extract table data
            table_data = tab.extract()
            if table_data and len(table_data) > 1:  # At least header + 1 row
                # Convert to readable format
                table_text = "Table:\n"
                for row_idx, row in enumerate(table_data):
                    if row_idx == 0:  # Header
                        table_text += "Headers: " + " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                    else:
                        table_text += f"Row {row_idx}: " + " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                
                tables.append({
                    'index': tab_index,
                    'data': table_data,
                    'text': table_text,
                    'bbox': tab.bbox
                })
    except Exception as e:
        print(f"Error extracting tables: {e}")
    
    return tables

#process_pdf function is the heart of your multimodal RAG system 
## Process PDF
def process_pdf(pdf_path):
    """Process PDF and create vector store"""
    global all_docs, all_embeddings, image_data_store, table_data_store, vector_store

    all_docs = []
    all_embeddings = []
    image_data_store = {}
    table_data_store = {}

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # ✅ ensure file handle closes on Windows
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            print(f"Processing page {i+1}...")

            # Process text
            text = page.get_text()
            if text.strip():
                temp_doc = Document(page_content=text, metadata={"page": i+1, "type": "text"})
                text_chunks = splitter.split_documents([temp_doc])

                for chunk in text_chunks:
                    embedding = embed_text(chunk.page_content)
                    all_embeddings.append(embedding)
                    all_docs.append(chunk)

            # Process tables
            tables = extract_tables_from_page(page)
            for tab_index, table in enumerate(tables):
                try:
                    table_id = f"page_{i+1}_table_{tab_index}"
                    print(f"Found table: {table_id} ({len(table['data'])} rows)")

                    table_data_store[table_id] = table['data']

                    embedding = embed_text(table['text'])
                    all_embeddings.append(embedding)

                    table_doc = Document(
                        page_content=table['text'],
                        metadata={"page": i+1, "type": "table", "table_id": table_id}
                    )
                    all_docs.append(table_doc)

                except Exception as e:
                    print(f"Error processing table {tab_index} on page {i+1}: {e}")
                    continue

            # Process images
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")

                    if not image_bytes:
                        print(f"Skipping missing image {img_index} on page {i+1}")
                        continue

                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    if pil_image.width < 100 or pil_image.height < 100:
                        print(f"Skipping small image {img_index} on page {i+1} ({pil_image.width}x{pil_image.height})")
                        continue

                    image_id = f"page_{i+1}_img_{img_index}"
                    print(f"Found meaningful image: {image_id} ({pil_image.width}x{pil_image.height})")

                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data_store[image_id] = img_base64

                    embedding = embed_image(pil_image)
                    all_embeddings.append(embedding)

                    img_doc = Document(
                        page_content=f"Visual diagram from page {i+1} showing attention mechanism components",
                        metadata={"page": i+1, "type": "image", "image_id": image_id}
                    )
                    all_docs.append(img_doc)

                except Exception as e:
                    print(f"Error processing image {img_index} on page {i+1}: {e}")
                    continue

    # Build FAISS vector store after PDF is closed
    embeddings_array = np.array(all_embeddings)
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs]
    )
    print(f"\n✅ Processed {len(all_docs)} documents total")
    print(f"✅ Stored {len(image_data_store)} meaningful images")


# Agent Classes
class RetrievalAgent:
    def __init__(self):
        self.llm = get_llm()

    def retrieve_documents(self,query:str, k: int=4)-> List[Document] :
        """Unified retrieval using CLIP embeddings for both text and images."""
        # Embed query using CLIP
        query_embedding = embed_text(query)
        
        # Search in unified vector store
        results = vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=k
        )
        
        return results

class TextAnalysisAgent:
    def __init__(self):
        self.llm = get_llm()

    def analyze_text(self, query: str, text_docs: List[Document]) -> str:
        """Analyze Text documents and provide insights"""
        if not text_docs:
            return "No Text Document Found for analyze"
        
        text_content="\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])

        prompt = f"""
                Summarize concisely the key points from the following text to answer the query.

                Query: {query}

                Text:
                {text_content}

                Answer in 4–5 short sentences.
                Analyse:
                """


        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
class ImageAnalysisAgent:
    def __init__(self):
        self.llm = get_llm()

    def analyze_images(self, query: str, image_docs: List[Document]) -> str:
        """Analyze images using GPT-4V"""
        if not image_docs:
            return "No images found for analysis."
        
        content = [
            {
                "type": "text",
                "text": f"As an image analysis specialist, analyze the following images to answer this query: {query}\n\nImages:"
            }
        ]
        
        for doc in image_docs:
            image_id = doc.metadata.get("image_id")
            page_num = doc.metadata.get("page", "Unknown page")  

            if image_id and image_id in image_data_store:
                content.append({
                    "type": "text",
                    "text": f"\n[Image from page {page_num}]:"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data_store[image_id]}"
                    }
                })
        
        message = HumanMessage(content=content)
        response = self.llm.invoke([message])
        return response.content

class TableAnalysisAgent:
    def __init__(self):
        self.llm = get_llm()
    
    def analyze_tables(self, query: str, table_docs: List[Document]) -> str:
        """Analyze table data"""
        if not table_docs:
            return "No tables found for analysis."
        
        table_content = "\n\n".join([
            f"[Table from page {doc.metadata.get('page', 'Unknown')}]:\n{doc.page_content}"
            for doc in table_docs
        ])
    
        prompt = f"""
                Summarize the main insights from the following tables relevant to the query.

                Query: {query}

                Table Data:
                {table_content}

                Answer in 3–4 concise sentences.
                Analyse:
                """

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class SynthesisAgent:
    def __init__(self):
        self.llm = get_llm()

    def synthesize_findings(self, query: str, text_analysis: str, image_analysis: str, table_analysis: str) -> str:
        """Synthesize findings from all agents with concise summary output"""
        prompt = f"""
        As a synthesis specialist, combine the following analyses from different agents to provide a concise summary answer.

        Original Query: {query}

        Text Analysis:
        {text_analysis}

        Image Analysis:
        {image_analysis}

        Table Analysis:
        {table_analysis}

        Provide a clear, brief summary that:
        1. Directly answers the original query
        2. Integrates key insights from text, images, and tables
        3. Highlights the most important findings
        4. Is no longer than 4-5 sentences

        Summary:
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class UserProxyAgent:
    def __init__(self):
        self.llm = get_llm()
    
    def validate_answer(self, query: str, answer: str)->Dict[str,Any]:
        """simullate user validation of answer"""

        return{
            "approved":True,
            "feedback":"Answer look well structured",
            "needs_revsion":False
        }

#initialize agents
retrieval_agent=RetrievalAgent()
text_agent=TextAnalysisAgent()
image_agent=ImageAnalysisAgent()
table_agent=TableAnalysisAgent()
synthesis_agent=SynthesisAgent()
user_proxy=UserProxyAgent()

# Agent functions for the graph
def retrieval_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents and force-include images/tables if mentioned."""
    print("Retrieval Agent: Searching for relevant documents...")
    retrieved_docs = retrieval_agent.retrieve_documents(state["query"])

    q_lower = state["query"].lower()

    # Force include images if query mentions figure/image/diagram
    if any(word in q_lower for word in ["figure", "image", "diagram"]):
        for img_id in image_data_store.items():
            retrieved_docs.append(
                Document(
                    page_content=f"Image related to query from {img_id}",
                    metadata={"type": "image", "image_id": img_id}
                )
            )

    # Force include tables if query mentions table/dataset/data
    if any(word in q_lower for word in ["table", "dataset", "data chart"]):
        for table_id, table_data in table_data_store.items():
            # Convert table to readable text
            table_text = "\n".join([" | ".join(map(str, row)) for row in table_data])
            retrieved_docs.append(
                Document(
                    page_content=f"Table from {table_id}:\n{table_text}",
                    metadata={"type": "table", "table_id": table_id}
                )
            )

    state["retrieved_docs"] = retrieved_docs
    state["current_agent"] = "retrieval"
    state["messages"].append(f"Retrieved {len(retrieved_docs)} relevant documents")

    return state

def text_analysis_node(state:AgentState)->AgentState:
    """Analyze text Documents"""
    print(" Text Analysis Agent: Analyzing text content...")
    text_docs=[doc for doc in state["retrieved_docs"] if doc.metadata.get("type") =="text"]
    text_analysis=text_agent.analyze_text(state["query"], text_docs)

    state["text_analysis"] = text_analysis
    state["current_agent"] = "text_analysis"
    state["messages"].append(f"Analyzed {len(text_docs)} text documents")
    
    return state

def image_analysis_node(state: AgentState) -> AgentState:
    """Analyze images"""
    print(" Image Analysis Agent: Analyzing visual content...")
    image_docs = [doc for doc in state["retrieved_docs"] if doc.metadata.get("type") == "image"]
    image_analysis = image_agent.analyze_images(state["query"], image_docs)
    
    state["image_analysis"] = image_analysis
    state["current_agent"] = "image_analysis"
    state["messages"].append(f"Analyzed {len(image_docs)} images")
    
    return state

def table_analysis_node(state: AgentState) -> AgentState:
    """Analyze tables"""
    print(" Table Analysis Agent: Analyzing tabular data...")
    table_docs = [doc for doc in state["retrieved_docs"] if doc.metadata.get("type") == "table"]
    table_analysis = table_agent.analyze_tables(state["query"], table_docs)
    
    state["table_analysis"] = table_analysis
    state["current_agent"] = "table_analysis"
    state["messages"].append(f"Analyzed {len(table_docs)} tables")
    
    return state

def synthesis_node(state: AgentState) -> AgentState:
    """Synthesize all findings"""
    print("🔧 Synthesis Agent: Combining all analyses...")
    final_answer = synthesis_agent.synthesize_findings(
        state["query"],
        state["text_analysis"],
        state["image_analysis"], 
        state["table_analysis"]
    )
    
    state["final_answer"] = final_answer
    state["current_agent"] = "synthesis"
    state["messages"].append("Synthesized final comprehensive answer")
    
    return state

def user_proxy_node(state: AgentState) -> AgentState:
    """User proxy validation"""
    print("👤 User Proxy: Validating final answer...")
    validation = user_proxy.validate_answer(state["query"], state["final_answer"])
    
    state["task_complete"] = validation["approved"]
    state["current_agent"] = "user_proxy"
    state["messages"].append(f"User validation: {validation['feedback']}")
    
    return state

# Create the multi-agent workflow
def create_multiagent_workflow():
    """create langgraph workflow"""
    workflow=StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("text_analysis", text_analysis_node)
    workflow.add_node("image_analysis", image_analysis_node)
    workflow.add_node("table_analysis", table_analysis_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("user_proxy", user_proxy_node)

    #Add edges
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "text_analysis")
    workflow.add_edge("text_analysis", "image_analysis")
    workflow.add_edge("image_analysis", "table_analysis")
    workflow.add_edge("table_analysis", "synthesis")
    workflow.add_edge("synthesis", "user_proxy")
    workflow.add_edge("user_proxy", END)

    return workflow.compile()

#Main Execution Function
def multiagent_pdf_rag_pipeline(query: str, pdf_path: str = None):

    """Execute multiagent RAG-pipline"""
    #proceesed pdf if provided
    if pdf_path and vector_store is None:
            print(f"📄 Processing PDF: {pdf_path}")
            process_pdf(pdf_path)
    
    # Create workflow
    app=create_multiagent_workflow()

    # Initial state
    initial_state = AgentState(
        messages=[],
        query=query,
        retrieved_docs=[],
        text_analysis="",
        image_analysis="",
        table_analysis="",
        final_answer="",
        current_agent="",
        task_complete=False
    )
    
    print(f"\n Starting multi-agent pipeline for query: {query}")
    print("="*80)
    
    # Execute the workflow
    result = app.invoke(initial_state)
    
    print("\n Execution Summary:")
    for message in result["messages"]:
        print(f"   {message}")
    
    return result["final_answer"]


