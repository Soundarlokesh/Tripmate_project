# TripMate AI Agent

An agentic travel assistant. A Groq-hosted LLM, orchestrated by LangGraph, decides per question whether to search a
FAISS-backed destination guide, fetch weather from Open-Meteo, use both, or answer directly (greeting, clarification,
out-of-scope decline). Every run produces a structured trace of what actually executed.

> **Verification status (read first).** The offline test suite passes (140 passed, 5 live tests skipped). The
> sandbox this was built in blocks Groq, Hugging Face and Open-Meteo, so **live Groq tool selection, real Hugging Face
> embeddings and live Open-Meteo calls have not been run**. Those paths are implemented and unit-tested against test
> doubles and mocked HTTP only. See [Verification](#verification) for how to run them.

## Problem statement

Travel information is spread across static FAQ pages. TripMate answers natural-language questions on visa/entry notes,
best time to visit, local customs, packing, safety and weather by choosing tools dynamically rather than following a
fixed script or keyword routing.

## Features

- LLM-driven tool choice through Groq tool calling: guide only, weather only, both, or none.
- RAG over the supplied 4-city dataset (Tokyo, Reykjavik, Bangkok, Barcelona), persisted FAISS index.
- Weather via Open-Meteo: real forecasts (to 15 days ahead), observed past weather, and clearly labelled
  `climate_reference` data for month-only or far-future requests. Never invents a forecast when the API fails.
- Clarifying questions, honest out-of-scope handling, per-session conversation memory.
- Tool name and argument validation, contained tool errors, step limit, LLM-failure handling.
- Structured JSON logging plus a per-request trace (`AgentResult.trace`).

## Technology stack

| Concern | Choice |
|---|---|
| Language | Python 3.10+ (developed and tested on 3.13) |
| LLM | Groq via `langchain-groq` (`ChatGroq`), default `llama-3.3-70b-versatile` |
| Orchestration | LangGraph `StateGraph` with conditional edges |
| Vector store | FAISS (`faiss-cpu`) through `langchain-community` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Validation / config | Pydantic, python-dotenv |
| HTTP | httpx |
| Tests | pytest |

**Why these:** Groq gives fast hosted open-weight models with OpenAI-style tool calling, so tool selection is real model
behaviour. LangChain supplies the `ChatGroq`, tool and embedding abstractions; LangGraph makes the agent loop an explicit,
inspectable graph (routing, step limit, per-session checkpointing) instead of a hand-rolled `while` loop. FAISS is a
small local index that needs no server and persists to disk. MiniLM-L6 is small (~90 MB, 384-dim, CPU-friendly) and good
enough for short English passages; it is swappable via `EMBEDDING_MODEL`.

**Model compatibility:** `llama-3.3-70b-versatile` is marked as tool-calling capable in langchain-groq's model profile,
and `build_llm` refuses models whose profile says otherwise. That profile is not a live check; confirm with
`tests/test_live.py`. Other tool-capable Groq models (e.g. `openai/gpt-oss-20b`) can be set through `GROQ_MODEL`, untested here.

## Project structure

```
app/
  main.py            CLI entry point + wiring of live components
  config.py          Settings (env / .env), validation
  logging_config.py  JSON log formatter
  agent/             graph.py (assembly + TripMateAgent), nodes.py, routing.py, state.py, prompts.py
  tools/             destination_search.py, weather.py (provider interface + Open-Meteo), registry.py (LangChain tools)
  rag/               ingestion.py, chunking.py, embeddings.py, vector_store.py
  schemas/           responses.py (WeatherResult, TraceEvent, AgentResult)
data/destinations/   supplied dataset (source of truth); data/index/ is generated and git-ignored
scripts/             ingest_data.py, calibrate_threshold.py, record_traces.py
tests/               offline suite + fakes.py (test doubles) + test_live.py (opt-in)
docs/                architecture.md, requirements_checklist.md
examples/            sample_traces.jsonl
```

## Installation

```bash
git clone <your-repo-url> tripmate-ai-agent && cd tripmate-ai-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned; `pip check` is clean
# Optional, smaller CPU-only torch before the line above:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
cp .env.example .env                   # then set GROQ_API_KEY
```

Environment variables are documented in `.env.example`. `GROQ_API_KEY` is read only from the environment/`.env` (never
committed) and is only required when the live LLM is created; tests need no key.

## Ingest the dataset

```bash
python -m scripts.ingest_data
```

Reads `data/destinations/*.txt`, creates one chunk per section (20 chunks), embeds them (first run downloads the
MiniLM model, ~90 MB, then it is cached), and writes `data/index/destinations/{index.faiss,index.pkl,manifest.json}`.
No network/model? `EMBEDDING_PROVIDER=offline-hash python -m scripts.ingest_data` builds an index with a deterministic
lexical stand-in (not semantic; for tests and demos only).

## Run

```bash
python -m app.main                                   # interactive
python -m app.main --query "What are the local customs in Tokyo?" --show-trace
python -m scripts.calibrate_threshold                # check RELEVANCE_THRESHOLD for your embedding model
```

Example queries: "What are the local customs in Tokyo?", "What will the weather be like in Tokyo on 2026-10-14?",
"I'm visiting Tokyo next month. What should I pack?", "What should I pack for my holiday?", "Book a flight to Tokyo."

## How the workflow works

```
START -> agent -> route_after_agent -> tools -> agent -> ... -> respond -> END
                        |-> limit -> END     (step budget spent)
                        |-> END              (LLM call failed)
```

- **agent**: system prompt (scope, rules, today's date) + trimmed history -> `ChatGroq` with both tools bound.
- **route_after_agent** (`agent/routing.py`): tool calls -> `tools`; none -> `respond`; budget spent -> `limit`; LLM error -> end.
- **tools**: executes only registered tools, validates arguments against the Pydantic schema, catches exceptions,
  returns `ToolMessage`s (status `error` on failure) and records trace events.
- **respond / limit**: set the final text. `limit` also closes dangling tool calls so the saved conversation stays valid.
- Clarifying questions, out-of-scope declines and greetings are the LLM answering without tool calls, guided by the system
  prompt. There is no keyword routing anywhere in the code.
- State (`agent/state.py`): `messages`, `request_id`, `steps`, `trace`, `status`, `final_response`. Tool results live in
  `messages`. History is kept per `session_id` (in memory) and trimmed to `MAX_HISTORY_MESSAGES`.

## Tool schemas (as sent to the LLM)

`search_destination_guide(query: str) -> list[str]`
```json
{"name": "search_destination_guide",
 "description": "Search the TripMate destination guide knowledge base. Covers ONLY these destinations: Bangkok, Barcelona, Reykjavik, Tokyo. Topics: visa and entry requirements, best time to visit, local customs, packing tips, safety and health. Use it for destination-specific facts and advice. Returns text chunks labelled '<City>, <Country> - <Section>'. It contains no live weather, prices, flights or bookings.",
 "parameters": {"type": "object", "required": ["query"],
   "properties": {"query": {"type": "string", "maxLength": 500,
     "description": "Natural-language question or topic, including the city name when known, e.g. 'local customs in Tokyo' or 'packing tips for Reykjavik in winter'."}}}}
```

`get_weather_forecast(city: str, date_or_month: str) -> dict`
```json
{"name": "get_weather_forecast",
 "description": "Get weather for a city on a specific date or in a month (Open-Meteo). Real forecasts exist only for dates from today up to 15 days ahead; past dates return observed weather. Month-only requests and dates further ahead return a labelled 'climate_reference' built from last year's observations, which is NOT a forecast. Needs both a city and a date or month; do not call it if either is unknown. On status 'error', no weather data is available.",
 "parameters": {"type": "object", "required": ["city", "date_or_month"],
   "properties": {
     "city": {"type": "string", "maxLength": 100, "description": "City name only, e.g. 'Tokyo' (no country or extra words)."},
     "date_or_month": {"type": "string", "maxLength": 40, "description": "A specific date as YYYY-MM-DD, or a month as YYYY-MM or a month name such as 'November'. Convert relative phrases like 'next month' or 'this Friday' using today's date."}}}}
```
(The destination list in the description is generated from the loaded index.)

Weather result (`schemas/responses.py::WeatherResult`), e.g.
`{"status":"ok","city":"Tokyo","source":"open-meteo","is_live":true,"data_kind":"forecast","conditions":"mild, mainly clear","temp_range_c":[11.0,18.5],"avg_high_c":18.5,"avg_low_c":11.0,"total_precipitation_mm":0.0,"rainy_days":0,"days_with_data":1,"limitations":[]}`.
Failures return `{"status":"error","error_type":"service_unavailable","message":"weather service timed out after 10s",...}` with no weather values.
Test doubles report `"source":"mock-test-data","is_live":false`.

## How FAISS retrieval works

1. Query -> embedding (same model as the index) -> FAISS `IndexFlatL2` over unit vectors (`normalize_L2=True`), so
   cosine similarity = `1 - d/2`.
2. If the query names an indexed destination (case-insensitive match against the manifest's list), results are restricted to it with a metadata filter.
3. Top `TOP_K` chunks (default 3) with similarity >= `RELEVANCE_THRESHOLD` are returned as
   `"<City>, <Country> - <Section>: <text> (source: <file>)"`.
4. Nothing relevant -> empty list; the LLM-facing tool then returns a note saying so and listing the supported cities.
   A query that names no supported city also gets that note, so the LLM can say "not covered" instead of reusing another city's text.

**Chunking:** sections in the dataset are 230-450 characters, so `RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)`
keeps each section as one chunk (one topic per vector); it only splits longer sections added later. Metadata kept: destination, country, section, source file.

**Relevance threshold.** Scores were measured on the actual dataset with the *offline-hash* stand-in: relevant top hits
scored 0.26-0.49, off-topic queries 0.0-0.13, so its default is 0.10 (lets lower-ranked but useful chunks through; a lexical
model is crude and still passes some generic words). **The default for the Hugging Face model (0.25) is a provisional value I could
not measure** because the model could not be downloaded here. Run `python -m scripts.calibrate_threshold` with your embeddings and set `RELEVANCE_THRESHOLD` if it suggests a different value.

**Index safety:** LangChain's FAISS loader unpickles the docstore, so `allow_dangerous_deserialization=True` is required. The index is only loaded from the configured local path after its SHA-256 hashes
and embedding id match `manifest.json`. Do not point `FAISS_INDEX_PATH` at an index from an untrusted source (a manifest can be forged by someone who can write to the directory).

## Example traces

Real output of the graph with a **scripted LLM test double and mock weather** (not live Groq; full set in
`examples/sample_traces.jsonl`; regenerate with `python -m scripts.record_traces`, or run it with `--live` against real services).

Multi-tool: "I'm visiting Tokyo next month. What should I pack?"
```json
{"node":"input","event":"request_received","summary":"I'm visiting Tokyo next month. What should I pack?"}
{"node":"agent","event":"llm_call","status":"ok","duration_ms":1.7,"summary":"model requested tools: search_destination_guide, get_weather_forecast"}
{"node":"tools","event":"tool_call","tool":"search_destination_guide","arguments":{"query":"packing tips for Tokyo"},"purpose":"Fetch destination-specific guidance from the knowledge base","status":"success","duration_ms":1.5,"summary":"2 chunk(s): Tokyo, Japan - Packing Tips; Tokyo, Japan - Local Customs"}
{"node":"tools","event":"tool_call","tool":"get_weather_forecast","arguments":{"city":"Tokyo","date_or_month":"2026-11"},"purpose":"Fetch weather data for the requested city and date or month","status":"success","duration_ms":1.1,"summary":"climate_reference from mock-test-data (live=False), cool, some wet days, temp_range_c=[4.0, 12.0]"}
{"node":"agent","event":"llm_call","status":"ok","duration_ms":0.4,"summary":"model answered directly (no tool calls)"}
{"node":"respond","event":"final_response","status":"answered","summary":"1459 chars"}
```

Out of scope: "Book a flight to Tokyo." (no tools, honest decline)
```json
{"node":"input","event":"request_received","summary":"Book a flight to Tokyo."}
{"node":"agent","event":"llm_call","status":"ok","summary":"model answered directly (no tool calls)"}
{"node":"respond","event":"final_response","status":"answered","summary":"91 chars"}
```

Weather timeout: guide still used, weather error captured, no weather invented
```json
{"node":"tools","event":"tool_call","tool":"get_weather_forecast","arguments":{"city":"Tokyo","date_or_month":"2026-10-12"},"status":"error","summary":"error (service_unavailable): weather service timed out after 10s","error":"weather service timed out after 10s"}
```

Each event also carries `request_id`. The same events are logged as JSON lines (logger `tripmate.trace`, stderr).
User queries (first 200 chars) appear in the trace; prompts, model replies and API keys are not logged.

## Tests

```bash
pytest -v                                  # offline: no API keys, no network
RUN_LIVE_TESTS=1 pytest tests/test_live.py -v -s   # live Groq/Open-Meteo/HF; needs a built index
```

Offline suite: RAG (ingestion, chunking, FAISS build/persist/load/tamper, retrieval, unknown destination), weather
(forecast/archive/climate reference, invalid input, HTTP timeout/failure/invalid payload via `httpx.MockTransport`), routing,
LangGraph scenarios A-H, and an end-to-end packing flow over the real dataset. The LLM there is a scripted double, so
these tests show the graph, tools and tracing work, **not** that a live model picks the right tools.

## Error handling

| Situation | Behaviour |
|---|---|
| Empty / over-long user input | `invalid_input` result, no LLM call |
| Empty search query, bad weather args | rejected by schema validation; error `ToolMessage` returned to the LLM |
| Unknown tool name from the model | rejected, never executed, traced as `rejected_unknown_tool` |
| Unknown city / invalid date / API timeout / HTTP error / bad payload | `status: error` weather result; model told to say weather is unavailable |
| Unexpected exception in a tool | caught at the tool boundary, logged with traceback, reported as tool error |
| Missing index / tampered index / wrong embedding model | startup error with the command to rebuild |
| Malformed or missing source data | `IngestionError` naming the file |
| Groq failure (network, auth, rate limit) | friendly `llm_error` reply, trace records the error |
| Model keeps requesting tools | `MAX_GRAPH_STEPS` stops the run (`step_limit_reached`) |

## Limitations

- Live behaviour (Groq tool choice, HF embeddings, Open-Meteo responses) is unverified here; the Open-Meteo parameter names and endpoint rules come from my knowledge of its API, not a live call.
- Python 3.10 compatibility is declared but only 3.13 was tested.
- Destination detection for the FAISS filter is a name match against the indexed cities; spelling variants ("Reykjavík") fall back to unfiltered search.
- Unknown-destination handling relies on the coverage note plus the system prompt; a model could still ignore it.
- `climate_reference` is last year's observed weather, a crude guide. Geocoding takes the first match (no disambiguation of "Paris, Texas").
- Conversation memory is in-process (`MemorySaver`): lost on restart, not shared across workers, grows until trimmed per call.
- A malformed tool call from the model (Groq `tool_use_failed`) surfaces as an `llm_error` reply; no automatic retry.
- Dataset is a simplified assessment pack, not authoritative visa/travel advice.
- The Loom video is not recorded.

## Scalability considerations (discussion only)

- **Hundreds of cities:** flat FAISS is fine up to ~10^5-10^6 chunks; beyond that use IVF/HNSW indexes or a managed vector DB, keep the destination metadata filter (or a destination-resolver step first), embed in batches, and rebuild incrementally per changed file.
- **Redundant calls:** cache tool results (weather by city+date with a short TTL, retrieval by normalised query+destination), cache whole answers for identical queries, and use Groq prompt-caching-friendly stable prefixes.
- **LLM cost:** small model for routing/simple turns and a larger one only for synthesis, short tool outputs (top-k and chunk sizes), history trimming/summarisation, and rate-limit/backoff.
- **Many tools:** retrieve a shortlist of relevant tool descriptions per query (tool-RAG) or route through a small classifier before binding tools, keep descriptions short, and group tools into sub-agents.

## Future improvements

Live-measured threshold and retrieval evaluation set; retry on malformed tool calls; streaming responses; persistent
checkpointer (SQLite/Redis); a hybrid BM25+vector retriever; a small HTTP API/UI; weather caching; LangSmith or OpenTelemetry tracing.
