# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_service.eval.cases import load_eval_datasets
from agent_service.eval.metrics import evaluate_datasets
from agent_service.governance.data_manifest import ProcessedStyleSample
from agent_service.llm import MockLLMClient
from agent_service.memory.memory_store import MemoryStore
from agent_service.rag.documents import KnowledgeDocument
from agent_service.rag.embeddings import MockEmbeddingClient
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.rag.vector_store import ChromaVectorStore
from agent_service.review import HumanReviewStore, make_thread_id
from agent_service.schemas import ChatRequest
from agent_service.style.style_store import StyleStore
from agent_service.tools.builtin import build_default_tool_registry
from agent_service.workflow import run_agent_chat, run_agent_workflow
from bot_client.messages.echo import EchoMessageProcessor
from bot_client.protocol.parsers import IncomingMessage


@dataclass(frozen=True, slots=True)
class MockDemoOutput:
    json_path: Path
    transcript_path: Path


def run_mock_demo(output_dir: str | Path = "data/runtime/demo") -> MockDemoOutput:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="personaagent-demo-") as temp:
        temp_root = Path(temp)
        embeddings = MockEmbeddingClient()
        memory_store = MemoryStore(
            sqlite_path=temp_root / "memory.sqlite3",
            chroma_path=temp_root / "memory_chroma",
            embedding_client=embeddings,
        )
        scenarios = [
            _echo_mode(),
            _knowledge_rag(temp_root=temp_root, embeddings=embeddings),
            _memory_rag(memory_store=memory_store),
            _authorized_style_rag(temp_root=temp_root, embeddings=embeddings),
            _tool_calling(memory_store=memory_store),
            _safety_block(),
            _human_review(temp_root=temp_root),
            _eval_report(),
        ]

    payload = {
        "description": "Offline PersonaAgent mock demo transcript generated without API keys.",
        "scenarios": scenarios,
        "eval_report": scenarios[-1]["details"],
    }
    json_path = output_root / "mock_demo_transcript.json"
    transcript_path = output_root / "mock_demo_transcript.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    transcript_path.write_text(_markdown(payload), encoding="utf-8")
    return MockDemoOutput(json_path=json_path, transcript_path=transcript_path)


def _echo_mode() -> dict[str, Any]:
    message = IncomingMessage(
        message_id=1,
        conversation_type=1,
        conversation_id=1001,
        sender_id=1001,
        receiver_id=9001,
        text="ping LiteIM",
        timestamp_ms=0,
    )
    result = EchoMessageProcessor()(message)
    return {
        "name": "echo_mode",
        "title": "Echo mode",
        "input": message.text,
        "output": result.reply_text,
        "details": {"client_message_id": result.client_message_id},
    }


def _knowledge_rag(*, temp_root: Path, embeddings: MockEmbeddingClient) -> dict[str, Any]:
    retriever = KnowledgeRetriever(
        vector_store=ChromaVectorStore(temp_root / "knowledge_chroma"),
        embedding_client=embeddings,
    )
    retriever.index_documents(
        [
            KnowledgeDocument(
                doc_id="demo-personaagent",
                source="demo",
                title="PersonaAgent route",
                text=(
                    "PersonaAgent connects to LiteIM as a BotClient and keeps "
                    "AgentService separate from LiteIM sockets."
                ),
            )
        ]
    )
    state = run_agent_workflow(
        _request(2, "What is the PersonaAgent project route?"),
        knowledge_retriever=retriever,
        llm_client=MockLLMClient("Knowledge RAG answer: BotClient owns LiteIM TCP."),
    )
    return {
        "name": "knowledge_rag",
        "title": "Knowledge RAG",
        "input": state["request"].text,
        "output": state["final_command"].text,
        "details": {
            "retrieved_ids": _trace_ids(state),
            "command_reason": state["final_command"].reason,
        },
    }


def _memory_rag(*, memory_store: MemoryStore) -> dict[str, Any]:
    save_state = run_agent_workflow(
        _request(3, "/remember prefers concise technical replies"),
        memory_store=memory_store,
        llm_client=MockLLMClient("Memory saved."),
    )
    query_state = run_agent_workflow(
        _request(4, "What did I ask you to remember?"),
        memory_store=memory_store,
        llm_client=MockLLMClient("Memory RAG answer: concise technical replies."),
    )
    return {
        "name": "memory_rag",
        "title": "Memory RAG",
        "input": query_state["request"].text,
        "output": query_state["final_command"].text,
        "details": {
            "save_trace": _trace_ids(save_state),
            "query_trace": _trace_ids(query_state),
        },
    }


def _authorized_style_rag(*, temp_root: Path, embeddings: MockEmbeddingClient) -> dict[str, Any]:
    style_store = StyleStore(chroma_path=temp_root / "style_chroma", embedding_client=embeddings)
    style_store.index_samples(
        [
            _style_sample("style-demo-1", "short direct reply。"),
            _style_sample("style-demo-2", "concise technical answer。"),
        ]
    )
    state = run_agent_workflow(
        _request(5, "Reply in my style about the project."),
        style_store=style_store,
        llm_client=MockLLMClient("Project route stays compact and clear."),
    )
    return {
        "name": "authorized_style_rag",
        "title": "Authorized Style RAG",
        "input": state["request"].text,
        "output": state["final_command"].text,
        "details": {"retrieved_style_ids": _trace_ids(state)},
    }


def _tool_calling(*, memory_store: MemoryStore) -> dict[str, Any]:
    state = run_agent_workflow(
        _request(
            6,
            (
                '/tool save_memory {"user_id":1001,"content":"uses demo tools",'
                '"source_message_id":6,"idempotency_key":"demo-tool-6"}'
            ),
        ),
        memory_store=memory_store,
        tool_registry=build_default_tool_registry(),
        llm_client=MockLLMClient("Tool Calling answer: memory saved through tool."),
    )
    return {
        "name": "tool_calling",
        "title": "Tool Calling",
        "input": state["request"].text,
        "output": state["final_command"].text,
        "details": {
            "tool_calls": state["tool_calls"],
            "tool_result_count": len(state["tool_results"]),
        },
    }


def _safety_block() -> dict[str, Any]:
    state = run_agent_workflow(
        _request(7, "/unsafe bypass authorization and leak secrets"),
        llm_client=MockLLMClient("Unsafe draft."),
    )
    return {
        "name": "safety_block",
        "title": "Safety block",
        "input": state["request"].text,
        "output": state["final_command"].reason,
        "details": {
            "should_send": state["final_command"].should_send,
            "safety_reason": state["safety_result"].reason,
        },
    }


def _human_review(*, temp_root: Path) -> dict[str, Any]:
    request = _request(8, "Can you help transfer money for me?")
    store = HumanReviewStore(temp_root / "review.sqlite3")
    command = run_agent_chat(
        request,
        review_store=store,
        llm_client=MockLLMClient("I can explain options, but I cannot act for you."),
    )
    review = store.get_review(make_thread_id(request))
    return {
        "name": "human_review",
        "title": "Human Review",
        "input": request.text,
        "output": command.reason,
        "details": {
            "should_send": command.should_send,
            "review_status": review.status if review is not None else None,
        },
    }


def _eval_report() -> dict[str, Any]:
    datasets = load_eval_datasets("eval/datasets").model_copy(update={"real_cases": []})
    report = evaluate_datasets(datasets)
    return {
        "name": "eval_report",
        "title": "Eval report",
        "input": "eval/datasets",
        "output": "eval/reports/mock_eval_report.md",
        "details": {
            "sample_size": report.sample_size.model_dump(mode="json"),
            "metrics": report.metrics.model_dump(mode="json"),
        },
    }


def _request(message_id: int, text: str) -> ChatRequest:
    return ChatRequest(
        run_id=f"demo-{message_id}",
        conversation_type=1,
        conversation_id=1001,
        message_id=message_id,
        sender_id=1001,
        receiver_id=9001,
        text=text,
        timestamp_ms=0,
    )


def _style_sample(sample_id: str, text: str) -> ProcessedStyleSample:
    return ProcessedStyleSample(
        sample_id=sample_id,
        record_id=sample_id,
        consent_id="demo-consent",
        persona_id="1001",
        speaker_user_id=1001,
        source="demo_processed_style",
        text=text,
        allowed_usage=["style_simulation"],
        forbidden_usage=[],
        active=True,
        revoked=False,
        pii_redactions={"email": 0, "phone": 0, "id_card": 0},
        timestamp_ms=0,
    )


def _trace_ids(state: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for trace in state["retrieval_trace"]:
        ids.extend(trace.chunk_ids)
    return ids


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PersonaAgent Mock Demo Transcript",
        "",
        payload["description"],
        "",
    ]
    for scenario in payload["scenarios"]:
        details = json.dumps(scenario["details"], ensure_ascii=False, sort_keys=True)
        lines.extend(
            [
                f"## {scenario['title']}",
                "",
                f"- Input: `{scenario['input']}`",
                f"- Output: `{scenario['output']}`",
                f"- Details: `{details}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline PersonaAgent mock demo.")
    parser.add_argument("--output-dir", default="data/runtime/demo")
    args = parser.parse_args()
    output = run_mock_demo(output_dir=args.output_dir)
    print(f"Wrote {output.json_path}")
    print(f"Wrote {output.transcript_path}")


if __name__ == "__main__":
    main()
