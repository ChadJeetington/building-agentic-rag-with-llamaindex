import sys
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

# Repo root on path so `helper` resolves (e.g. notebooks import utils from a subfolder).
_REPO_ROOT = Path(__file__).resolve().parent
_r = str(_REPO_ROOT)
if _r not in sys.path:
    sys.path.insert(0, _r)

from helper import get_anthropic_api_key  # noqa: E402

SONNET_MODEL = "claude-sonnet-4-6"
HF_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _configure_llama_settings() -> None:
    api_key = get_anthropic_api_key()
    Settings.llm = Anthropic(model=SONNET_MODEL, api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBED_MODEL)


def get_router_query_engine(pdf_path: str) -> RouterQueryEngine:
    """Build a RouterQueryEngine over a single PDF (MetaGPT lesson pattern)."""
    _configure_llama_settings()
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, show_progress=True)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=False,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions related to MetaGPT",
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True,
    )


def get_doc_tools(file_path: str, name: str):
    """Vector + summary tools for one document (L3/L4 lesson pattern)."""
    _configure_llama_settings()
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, show_progress=True)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=False,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to the paper {name}"
        ),
    )
    vector_tool = QueryEngineTool.from_defaults(
        name=f"vector_tool_{name}",
        query_engine=vector_query_engine,
        description=(
            f"Useful for retrieving specific details about the paper {name}"
        ),
    )
    return vector_tool, summary_tool
