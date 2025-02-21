import os
import json
import glob
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import dotenv
import gradio as gr
from gradio_pdf import PDF
import plotly.graph_objects as go
import plotly.graph_objects as go
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.util import cos_sim
from sklearn.decomposition import PCA
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import click

TEXT_QA_SYSTEM_PROMPT = """ You are a renowned expert Question/Answer system.
Always answer the question using only the provided context, and no other information.

Some rules to follow:

Never reference the provided context in your response.
Avoid phrases like "According to the context..." or "The context states..." or expressions like "these lines."
If the provided context includes document pages, cite the relevant page(s) in your response.
"""

DEFAULT_TEXT_QA_PROMPT_TMPL = """ The context is below, between two dashed lines.
------------------
{context_str}
------------------
Given this context and no other information, answer the question.

If applicable, cite the relevant page(s) and extract from the document in your response.

Question: {query_str}
Answer:
"""

DEFAULT_TEMPLATE = """
Given a conversation (between a human and an assistant) and a new message from the human,
rewrite this message as a standalone question that captures the relevant context from previous exchanges, prioritizing recent interactions.
Only provide the standalone question as your response.

History of the conversation: {history}

<New Message>
{question}

<Standalone Question>
"""

def plot_rag(points, highlighted_points, query_point, size=10):
    # Convert lists to NumPy arrays for easier indexing
    points = np.array(points, dtype=object)
    highlighted_points = np.array(highlighted_points, dtype=object)

    # Extract positions and labels correctly
    x_points = [point[0][0] for point in points]
    y_points = [point[0][1] for point in points]
    z_points = [point[0][2] for point in points]
    text_points = [p[1] for p in points]
    pages_points = [p[2] for p in points]

    x_highlighted = [point[0][0] for point in highlighted_points]
    y_highlighted = [point[0][0] for point in highlighted_points]
    z_highlighted = [point[0][0] for point in highlighted_points]
    text_highlighted = [p[1] for p in highlighted_points]
    pages_highlighted = [p[2] for p in highlighted_points]
    sizes = [float(point[3]) for point in highlighted_points]

    # Query point extraction
    x_query = query_point[0][0]
    y_query = query_point[0][1]
    z_query = query_point[0][2]
    text_query = query_point[1]

    # Create figure
    fig = go.Figure()

    # Regular points with white outline
    fig.add_trace(go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode="markers",
        marker=dict(
            size=size,
            color='blue',
            opacity=0.3,
            line=dict(
                color='white',  # White outline
                width=2  # Outline thickness
            )
        ),
        hovertemplate=[f'{text}<extra>p.{page}</extra>' for text, page in zip(text_points, pages_points)],
    ))

    # Highlighted points with white outline
    fig.add_trace(go.Scatter3d(
        x=x_highlighted,
        y=y_highlighted,
        z=z_highlighted,
        mode="markers",
        marker=dict(
            size=[(s*size) for s in sizes],
            color='red',
            opacity=0.8,
            line=dict(
                color='white',  # White outline
                width=3  # Thicker outline for emphasis
            )
        ),
        hovertemplate=[f'{text}<extra>p.{page}</extra>' for text, page in zip(text_highlighted, pages_highlighted)],
    ))


    # Query point with distinct styling
    fig.add_trace(go.Scatter3d(
        x=[x_query],
        y=[y_query],
        z=[z_query],
        mode="markers",
        marker=dict(
            size=size,  # Larger size for prominence
            color='limegreen',  # Green to stand out
            opacity=1,
            line=dict(
                color='white',  # Bold white outline
                width=4  # Thickest outline
            )
        ),
        customdata=[text_query],
        hovertemplate='%{customdata}<extra>Recherche</extra>',
    ))

    # Red lines connecting query point to all highlighted points
    for x_h, y_h, z_h in zip(x_highlighted, y_highlighted, z_highlighted):
        fig.add_trace(go.Scatter3d(
            x=[x_query, x_h],  # Start from query, end at highlighted
            y=[y_query, y_h],
            z=[z_query, z_h],
            mode="lines",
            line=dict(
                color='red',
                width=2
            ),
            hoverinfo="skip"  # Avoid unnecessary hover text on lines
        ))

    # Layout settings
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=440,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

def softmax(x):
  return np.exp(x)/sum(np.exp(x))

def process_pdf(pdf_path, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    loader = PyPDFLoader(pdf_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)

    # Split pages into small chunks
    chunks = text_splitter.split_documents(pages)

    # Create vector store with small chunks
    vector_store = InMemoryVectorStore.from_documents(
        chunks, 
        OpenAIEmbeddings()
    )

    # Dirty way to get all embeddings, text and page manually
    vector_store.dump("tmp.json")
    with open("tmp.json", "r") as f:
        data = json.load(f)
    os.remove("tmp.json")
    ids = list(data.keys())
    id_embedding_dict = {
        id_: {
            "id": data[id_]["metadata"]["page_label"],
            "vector": data[id_]["vector"],
            "text": data[id_]["text"],
            "page": data[id_]["metadata"]["page_label"]
        } for id_ in ids
    }

    chunks = []
    ids = []
    embeddings = []

    for id_, value in tqdm(id_embedding_dict.items()):
        ids.append(value["page"])
        chunks.append(value["text"])
        embeddings.append(value["vector"])

    print("about to return")

    return chunks, ids, embeddings

def query_and_format(query="", result_count=5, embeddings=None, chunks=None):
  """
  Format the 
  """
  query_embedding = OpenAIEmbeddings().embed_query(query)

  pca = PCA(n_components=3)
  embs_to_fit = embeddings + [query_embedding]
  reduced_embeddings = pca.fit_transform(embs_to_fit)

  # text processing
  cut_chunks = [chunk.replace("\n", "<br>").replace(". ", "<br>") for chunk in chunks]

  # fusing things together
  similarities = np.array(cos_sim(query_embedding, embeddings)[0])
  indexes = np.argsort(similarities)[-1 * result_count:]

  # size processing
  queried_similarities = softmax(similarities[indexes])
  min_val = min(queried_similarities)
  queried_similarities = [x - min_val for x in queried_similarities]
  queried_similarities = [3 * (1+50*x)**2 for x in queried_similarities]

  all_points = np.array(list(zip(reduced_embeddings[:-1], cut_chunks, ids)), dtype=object).copy()

  highlighted_points = [list(all_points[i]) for i in indexes]
  highlighted_points = [list(highlighted_points[i]) + [queried_similarities[i]] for i in range(len(highlighted_points))]

  query_point = (reduced_embeddings[-1], query)

  queried_ids = np.array(ids)[indexes]
  queried_chunks = np.array(chunks)[indexes]

  formatted_strings = [f"page n° {id_}:\n{chunk}" for id_, chunk in zip(queried_ids, queried_chunks)]

  return all_points, highlighted_points, query_point, formatted_strings

class MultiLLMChat:
    def __init__(self, system_prompt):
        self.current_model = "gpt"
        self.models = {
            "gpt3.5": self._init_gpt3_5(),
            "gpt4o": self._init_gpt4o(),
            "gemini": self._init_gemini(),
            "mistral": self._init_mistral()
        }

        # System prompt for consistent behavior across models
        if not system_prompt:
          self.system_prompt = """You are a helpful assistant analyzing a PDF document.
          Provide clear, concise answers based on the document content and maintain a friendly, professional tone."""
        else:
          self.system_prompt = system_prompt

    def _init_claude(self) -> ChatAnthropic:
        return ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7
        )


    def _init_gpt4o(self) -> ChatOpenAI:
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.7
        )

    def _init_gpt3_5(self) -> ChatOpenAI:
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )

    def _init_gemini(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7
        )

    def _init_mistral(self) -> ChatMistralAI:
        return ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.7
        )

    def switch_model(self, model_name: str):
        if model_name in self.models:
            self.current_model = model_name
            return f"Switched to {model_name}"
        return f"Invalid model name. Available models: {', '.join(self.models.keys())}"

    def get_response(self, message: str, history: List[Tuple[str, str]] = []) -> str:
        # Convert chat history to LangChain message format
        messages = [SystemMessage(content=self.system_prompt)]

        for human, ai in history:
            messages.append(HumanMessage(content=human))
            if ai:
                messages.append(AIMessage(content=ai))

        messages.append(HumanMessage(content=message))

        # Get response from current model
        try:
            if self.current_model == "gpt":
                with get_openai_callback() as cb:
                    response = self.models[self.current_model](messages)
                    print(f"Total Tokens: {cb.total_tokens}")
            else:
                response = self.models[self.current_model](messages)

            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

def chatbot_response(message: str, history: List[Tuple[str, str]], model_name: str, search_db: bool) -> Tuple[List[Tuple[str, str]], str]:
    # Switch model if requested
    if model_name != chat_manager.current_model:
        chat_manager.switch_model(model_name)

    message_to_send_to_the_llm = message

    new_plot = None
    if search_db:
      # rewrite the query using DEFAULT_TEMPLATE
      rephrased_quetion_prompt = DEFAULT_TEMPLATE.format(question=message, history=history)

      query = chat_manager.get_response(rephrased_quetion_prompt, history)

      # here is how to call the database and have a plot
      chunks_points, r_points, q_point, extracted_chunks = query_and_format(query, 5, embeddings, chunks)
      new_plot = plot_rag(chunks_points, r_points, q_point, size=5)

      # calling the LLM using the RAG template
      message_to_send_to_the_llm = DEFAULT_TEXT_QA_PROMPT_TMPL.format(context_str="\n\n".join(extracted_chunks), query_str=query)

    bot_message = chat_manager.get_response(message_to_send_to_the_llm, history)

    # Update history
    history = history + [(message, bot_message)]

    return history, "", new_plot # Return updated history and clear input

def build_app_gradio(pdf_path, available_model_choices):
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column(scale=1):
                pdf_viewer = PDF(label="Journal", value=pdf_path)

            with gr.Column(scale=1):
                model_selector = gr.Radio(
                    choices=available_model_choices,
                    value=available_model_choices[0],
                    label="Pick an LLM"
                )

                search_db = gr.Checkbox(
                    label="Look for a text chunk",
                    value=False
                )

                embedding_pca_plot = gr.Plot()

                # Chatbot with history
                chatbot = gr.Chatbot(
                    [],
                    height=500,
                    show_label=False,
                )

                # Message input
                msg = gr.Textbox(
                    label="Type your message here",
                    placeholder="Type your message here and press Enter",
                    show_label=False
                )

                # Clear button
                clear = gr.Button("Clear conversation")

                # Submit handler
                msg.submit(
                    chatbot_response,
                    [msg, chatbot, model_selector, search_db],
                    [chatbot, msg, embedding_pca_plot],
                )

                # Clear chat history
                def clear_chat():
                    return None

                clear.click(clear_chat, None, chatbot)

    return app

chat_manager = None
embeddings = None
chunks = None
ids = None

@click.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--chunk-size", default=1000, show_default=True, type=int, help="Size of chunks to process the PDF.")
def main(pdf_path, chunk_size):
    global chat_manager
    global embeddings
    global chunks
    global ids

    dotenv.load_dotenv()

    chunks, ids, embeddings = process_pdf(pdf_path, chunk_size)

    chat_manager = MultiLLMChat(TEXT_QA_SYSTEM_PROMPT)
    
    model_env_vars = {
        "gpt3.5": "OPENAI_API_KEY",
        "gpt4o": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "claude": "ANTHROPIC_API_KEY"
    }

    available_model_choices = [model for model, env_var in model_env_vars.items() if os.getenv(env_var)]

    print("Building the interface...")
    app = build_app_gradio(pdf_path, available_model_choices)
    
    print("Gradio app launched successfully")
    app.launch(debug=True, share=True)
    print("This line should print after Gradio closes")

if __name__ == "__main__":
    main()
