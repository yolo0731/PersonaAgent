# PersonaAgent Project Knowledge

## Project Positioning

PersonaAgent is an authorized persona-style AI Agent worker connected to LiteIM through a normal bot account.

The project is designed for interview presentation as an AI Agent system with clear engineering boundaries, retrieval, memory, authorized style data, safety checks, traceability, and evaluation.

## LiteIM Boundary

LiteIM is the C++ instant messaging system. PersonaAgent treats LiteIM as a black-box TCP/TLV endpoint.

PersonaAgent must not embed Python, LangGraph, LLM calls, or persona logic into the C++ LiteIM server.

AgentService must not directly access LiteIM MySQL or Redis. It does not hold a LiteIM TCP connection and does not send LiteIM packets directly.

BotClient is the only LiteIM network owner. It logs in as a normal LiteIM user, pulls offline messages, sends delivery/read ACKs, requests recent history through `HistoryRequest`, calls AgentService `/chat`, and sends final replies through `PrivateMessageRequest`.

## Runtime Flow

The online message flow is:

```text
Qt user client
  -> LiteIM C++ server
  -> Python BotClient
  -> FastAPI AgentService /chat
  -> LangGraph workflow
  -> AgentReplyCommand
  -> Python BotClient
  -> LiteIM PrivateMessageRequest
  -> Qt user client
```

AgentService returns structured commands only. BotClient executes send/no-send commands and records delivery execution state.

## LangGraph Workflow

The AgentService workflow keeps six core nodes:

1. `dialogue_policy`
2. `retrieve_context`
3. `tool_router`
4. `generate_reply`
5. `safety_check`
6. `finalize_reply`

`dialogue_policy` classifies the user message into intents such as smalltalk, knowledge question, memory update, memory query, style chat, history summary, unsafe, or command.

`retrieve_context` can load Knowledge RAG, Memory RAG, Authorized Style RAG, and recent LiteIM conversation context.

`generate_reply` builds a persona-aware prompt and calls the configured OpenAI-compatible LLM client.

`safety_check` blocks unsafe content, impersonation attempts, privacy leakage, unauthorized mimicry, and raw source leaks.

`finalize_reply` creates an idempotent `AgentReplyCommand` with a stable dedup key.

## Knowledge RAG

Knowledge RAG stores project knowledge documents under `data/knowledge_docs`.

AgentService loads markdown and text files from `KNOWLEDGE_DOCS_PATH`, splits them into chunks, embeds them with the configured embedding client abstraction, and writes them into provider/model-scoped Chroma knowledge collections under `CHROMA_PATH`.

Knowledge RAG is for stable project facts and interview explanations. It should contain architecture notes, module responsibilities, run commands, design tradeoffs, and known limitations.

Knowledge RAG is not the place for raw private chats or authorized style samples.

## Memory RAG

Memory RAG stores long-term conversation facts in PersonaAgent's own SQLite memory database and Chroma memory collection.

SQLite is the source of truth for memory records, active/inactive state, and listing. Chroma is the retrieval index.

Memory records are scoped by user id. One user's memories should not be retrieved for another user.

Examples of memory facts:

- 演示用户的生日是 1 月 1 日。
- 演示用户正在准备 C++ 后端和 AI Agent 岗位面试。
- 演示用户希望回复简短、自然、不要客服腔。

## Authorized Style RAG

Authorized Style RAG uses consented, processed, redacted chat records from the target speaker.

For the current demo:

- current user: 演示用户, WeChat name `当前用户`
- target persona: 示例伙伴, WeChat name `目标样本`
- bot LiteIM username: `persona_bot`
- bot display name: `示例伙伴`

`style_samples.local.jsonl` contains target-speaker style samples only.

`style_pairs.local.jsonl` contains `self_text -> target_reply` pairs so the Agent can understand conversational context before imitating response style.

Generated runtime replies are not indexed into the authorized style collection. They can be appended to runtime feedback files for later review, but they should not be treated as real target-speaker records.

## Safety Boundaries

PersonaAgent may simulate an authorized style, but it should not falsely claim to be the real human in high-risk identity contexts.

It should not expose private chat records, raw samples, source ids, file paths, secrets, API keys, or local database details.

It should not make real-world commitments on behalf of the real person, such as money transfers, legal commitments, medical advice, account operations, or promises that require the human's real consent.

It should avoid verbatim copying from authorized style samples. The style should be high-level and natural rather than direct text replay.

## Evaluation

The evaluation suite measures:

- RAG hit rate
- Memory hit rate
- style marker similarity
- verbatim leakage rate
- safety violation rate
- human review trigger rate
- latency
- token cost
- LiteIM integration facts

The default evaluation path is offline and deterministic. Real model evaluation requires a real API key and should be treated separately from unit tests.

## Current Honest Limitations

The current first-version system is not a full production compliance system.

It includes a local Human Review operations UI, but it is not a full production compliance or enterprise review platform.

It does not fine-tune a model.

Production retrieval defaults to real Gemini embeddings through `EMBEDDING_PROVIDER=gemini`, `EMBEDDING_MODEL=models/gemini-embedding-001`, and `GEMINI_API_KEY`. Deterministic mock embeddings remain only for tests and offline demos.

Knowledge RAG only knows documents that are placed under `data/knowledge_docs` and indexed into Chroma. If the directory has only `.gitkeep`, the knowledge collection will be empty.

## Interview Talking Points

The strongest technical point is boundary control: LiteIM remains a normal IM system, while PersonaAgent is an independent AI Agent worker connected through a normal bot account.

The second strongest point is retrieval separation: Knowledge RAG, Memory RAG, and Authorized Style RAG have different data sources, different governance rules, and different retrieval purposes.

The third strongest point is safety before side effects: AgentService only produces a command; BotClient performs the actual LiteIM send after safety, human review, deduplication, and idempotency checks.

The fourth strongest point is explainability: every workflow node records trace information, and the evaluation suite makes quality and safety measurable.
