"""Agent workflow helpers for LlamaIndex >= 0.11 (AgentWorkflow / FunctionAgent API)."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

from llama_index.core.agent import AgentWorkflow, FunctionAgent
from llama_index.core.llms import LLM
from llama_index.core.memory import BaseMemory
from llama_index.core.objects import ObjectRetriever
from llama_index.core.tools import BaseTool


def _patch_anthropic_to_thread_for_worker_threads() -> None:
    """Patch Anthropic's ``to_thread`` for Jupyter / ``nest_asyncio`` and thread pools.

    Sniffio only reports ``asyncio`` when ``asyncio.current_task()`` is set; with
    ``nest_asyncio`` that can be ``None`` even while a loop is running, and inside
    ``asyncio.to_thread`` workers there is no asyncio task. In those cases we still
    use ``asyncio.to_thread`` when ``get_running_loop()`` succeeds, otherwise
    ``anyio.to_thread.run_sync`` (non-asyncio backends).
    """
    try:
        import anthropic._utils._sync as anthropic_sync
    except ImportError:
        return
    if getattr(anthropic_sync, "_llamaindex_course_to_thread_patched", False):
        return

    import functools

    import anyio.to_thread
    import sniffio

    async def to_thread(
        func, /, *args, **kwargs
    ):  # type: ignore[no-untyped-def]
        use_asyncio_thread = False
        try:
            use_asyncio_thread = sniffio.current_async_library() == "asyncio"
        except sniffio.AsyncLibraryNotFoundError:
            # Jupyter + nest_asyncio: loop is running but ``current_task()`` is None,
            # so sniffio fails even on the main event-loop thread.
            try:
                asyncio.get_running_loop()
                use_asyncio_thread = True
            except RuntimeError:
                use_asyncio_thread = False
        if use_asyncio_thread:
            return await asyncio.to_thread(func, *args, **kwargs)
        return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))

    anthropic_sync.to_thread = to_thread  # type: ignore[assignment]
    anthropic_sync._llamaindex_course_to_thread_patched = True


def run_agent_workflow_sync(
    workflow: AgentWorkflow,
    user_msg: str,
    *,
    memory: Optional[BaseMemory] = None,
) -> Any:
    """Run ``AgentWorkflow`` from a sync context (e.g. Jupyter after ``nest_asyncio.apply()``)."""
    _patch_anthropic_to_thread_for_worker_threads()
    handler = workflow.run(user_msg=user_msg, memory=memory)
    return asyncio.get_event_loop().run_until_complete(handler)


def tool_calling_workflow_from_tools(
    tools: Sequence[BaseTool],
    llm: LLM,
    *,
    verbose: bool = True,
) -> AgentWorkflow:
    """Single tool-calling agent (replaces removed ``FunctionCallingAgentWorker`` + ``AgentRunner``)."""
    return AgentWorkflow.from_tools_or_functions(
        list(tools),
        llm=llm,
        verbose=verbose,
    )


def tool_calling_workflow_from_retriever(
    tool_retriever: ObjectRetriever,
    llm: LLM,
    *,
    system_prompt: Optional[str] = None,
    verbose: bool = True,
) -> AgentWorkflow:
    """Agent that pulls tools via an ``ObjectRetriever`` (L4 tool-retrieval pattern)."""
    agent = FunctionAgent(
        name="PaperAgent",
        description="Answers questions using tools retrieved for the query.",
        llm=llm,
        tool_retriever=tool_retriever,
        system_prompt=system_prompt,
    )
    return AgentWorkflow(agents=[agent], verbose=verbose)
