# Requirements checklist

Status key: **Verified** = behaviour exercised by the offline test suite (`pytest -v`: 129 passed, 5 live tests skipped).
**Partial** = implemented and tested only with a test double / mocked HTTP; the live path was not run.
**Pending** = not done. Test names are in `tests/`.

The build sandbox blocks Groq, Hugging Face and Open-Meteo, so nothing marked Partial has touched those services.

## Functional requirements (assessment)

| # | Requirement | Implementation | Verification | Status | Evidence / limitation |
|---|---|---|---|---|---|
| M1.1 | Accept natural-language query | `TripMateAgent.ask` (`agent/graph.py`) | `test_agent_graph.py` all scenarios; `test_invalid_input_never_reaches_the_llm` | Verified | Empty/over-long input rejected before the LLM |
| M1.2 | Dynamically decide which tools, per query | LLM tool calling via `ChatGroq.bind_tools`; no keyword routing (`agent/nodes.py`, `routing.py`) | Scenario tests with scripted LLM; `tests/test_live.py` for real model | Partial | Graph handles every decision path. **Live Groq tool selection not run**; `test_live.py` is ready (needs `RUN_LIVE_TESTS=1` + key) |
| M1.3 | Call tools in correct order if several | `tools` node executes calls in the order returned; chained rounds supported | `test_tools_can_be_chained_across_steps`, `test_packing_query_end_to_end` | Verified | Order is chosen by the (scripted) LLM |
| M1.4 | Synthesize one coherent response | `agent` node synthesises after `ToolMessage`s | `test_packing_query_end_to_end` (answer built from both results) | Partial | Synthesis quality of the real model unverified |
| M1.5 | Visible reasoning trace (tools, args, why) | `record()` in `nodes.py`; `TraceEvent`; `--show-trace`; JSON logs | `test_trace_is_consistent_logged_and_free_of_prompts`, `examples/sample_traces.jsonl` | Verified | "Why" = static per-tool `purpose` + step summaries; no model chain-of-thought is stored |
| M2 | `search_destination_guide(query) -> List[str]` over supplied data pack | `tools/destination_search.py`, `tools/registry.py`, `rag/*` | `test_destination_search.py` (34+ tests) | Partial | Real dataset, real FAISS. Tests use the offline lexical embeddings; **HF embeddings not run** |
| M3 | `get_weather_forecast(city, date_or_month) -> dict`, consistent schema | `tools/weather.py`, `schemas/responses.py::WeatherResult` | `test_weather.py` (53 tests) | Partial | Logic + HTTP layer verified against fakes/`MockTransport`; **live Open-Meteo not called** |
| M4 | At least one multi-tool query (packing) | System prompt + both tools bound | `test_packing_query_end_to_end`, `test_packing_question_can_use_both_tools_in_one_step` | Partial | Scripted LLM; live behaviour pending |
| M5 | Scope awareness (booking etc.) | System prompt; unknown tools rejected | `test_out_of_scope_request_is_declined_without_tools`, `test_hallucinated_tool_is_rejected_and_never_executed` | Partial | Decline wording comes from the LLM; the model's actual behaviour is unverified |

## Required scenarios (project brief)

| Scenario | Test | Status | Note |
|---|---|---|---|
| A Destination knowledge (Tokyo customs) | `test_destination_only_uses_only_the_guide_tool` | Partial | Grounded answer verified; tool choice scripted |
| B Weather only | `test_weather_only_passes_city_and_date_and_skips_the_guide` | Partial | Same |
| C Multi-tool packing, no fake monthly forecast | `test_packing_query_end_to_end`; weather month tests | Partial | Month request returns labelled `climate_reference` with limitation text |
| D Missing destination | `test_missing_destination_triggers_clarification_not_tools` | Partial | Clarification text is scripted |
| E Contextual follow-up, session-scoped | `test_follow_up_uses_conversation_context`, `test_sessions_do_not_share_context` | Partial | Context propagation verified; resolution by the model is not |
| F Out of scope | see M5 | Partial | |
| G Greeting | `test_greeting_needs_no_tools` | Partial | |
| H Weather timeout | `test_weather_timeout_is_reported_and_guide_still_used`, `test_http_timeout_is_reported` | Verified | Error captured, no weather values, guide still returned |

## Technical requirements

| Requirement | Implementation | Verification | Status | Evidence / limitation |
|---|---|---|---|---|
| Python 3.10+ | `requires-python`, no 3.11+ only syntax intended | Ran on 3.13 only | Partial | 3.10 not tested |
| Groq via LangChain | `build_llm` (`ChatGroq`) | Constructor + profile check run without network | Partial | No live call; model tool-calling support from langchain-groq profile only |
| LangGraph orchestration | `agent/graph.py` StateGraph, conditional edges | Graph tests | Verified | |
| FAISS vector store, persisted | `rag/vector_store.py` | persistence, reload, tamper, missing-index tests | Verified | |
| Hugging Face embeddings | `rag/embeddings.py` (`HuggingFaceEmbeddings`, MiniLM-L6) | Constructor failure path tested; model never loaded | Pending | Needs a network run of `python -m scripts.ingest_data` |
| Pinned compatible dependencies | `requirements.txt` | `pip check` clean on installed set | Verified | Top-level pins only |
| Model configurable by env | `GROQ_MODEL` | `test_config.py` | Verified | |
| No hard-coded secrets; `.env` ignored | `config.py`, `.gitignore`, `.env.example` | `test_api_key_is_not_in_repr` | Verified | |
| Index persisted, not in Git | `.gitignore` (`data/index/`) | inspection | Verified | |
| Safe index loading | manifest SHA-256 + embedding id check | tamper / mismatch tests | Verified | Pickle risk documented; manifest is not tamper-proof against a writer of the directory |
| Relevance threshold checked on dataset | `config.py`, `scripts/calibrate_threshold.py` | measured with offline embeddings | Partial | HF default 0.25 is **provisional/unmeasured** |
| Weather: Open-Meteo, timeouts, validation, mock labelled | `tools/weather.py` | `test_weather.py` | Partial | Live endpoints unverified |
| Offline tests need no key/network | `tests/conftest.py` | `pytest -v` run with no `GROQ_API_KEY`, network blocked | Verified | |
| Live tests separate | `tests/test_live.py` (skipped by default) | 5 skipped in the default run | Verified | Not executed |

## Non-functional requirements

| Requirement | Implementation | Verification | Status | Evidence |
|---|---|---|---|---|
| Layered design | tools / agent / rag / schemas packages | review | Verified | `weather.py` has no LangChain/LangGraph imports; `destination_search.py` depends only on the knowledge-base wrapper; LangChain tool definitions live in `registry.py` |
| Configuration management | `app/config.py` | `test_config.py` | Verified | |
| Structured logging of tool calls and decisions | JSON formatter, `tripmate.trace` logger | trace/log test | Verified | |
| Exception handling around tools | `_execute` in `nodes.py` | `test_unexpected_tool_exception_is_contained_and_traced` | Verified | |
| Input validation | `ask`, Pydantic arg schemas, weather validation | many tests | Verified | |
| Modular, readable code | ~1.7k lines app+scripts, ruff clean (F,E7,E9,B,UP) | `ruff check` | Verified | |

## Error handling (assessment list)

| Case | Verification | Status |
|---|---|---|
| Unknown/unsupported destination | `test_unknown_destination_adds_coverage_note_listing_supported_cities` | Partial (final wording depends on LLM) |
| Missing/incomplete weather data | `test_partial_data_is_reported_not_hidden`, `test_all_null_values_mean_no_data` | Verified |
| Ambiguous query | `test_missing_destination_triggers_clarification_not_tools` | Partial |
| Tool failure / timeout | H tests above | Verified |
| Out-of-scope | M5 | Partial |
| Malformed/empty input | `test_empty_query_is_rejected`, `test_invalid_input_never_reaches_the_llm`, weather invalid input tests | Verified |
| Graph iteration limit | `test_iteration_limit_stops_runaway_tool_loops`, `test_conversation_stays_valid_after_step_limit` | Verified |
| LLM failure | `test_llm_failure_returns_friendly_error` | Verified (simulated error) |

## Testing

| Requirement | Evidence | Status |
|---|---|---|
| Unit tests per tool | `test_destination_search.py`, `test_weather.py` | Verified |
| Tool-selection tests (single, multi, none) | `test_agent_graph.py` | Partial (scripted LLM; does not prove live selection) |
| Integration test, full multi-tool flow | `test_integration.py` | Partial (scripted LLM + mock weather; real dataset, ingest script, FAISS, graph, tools, trace) |
| Run result | `pytest -v`: 129 passed, 5 skipped (live) | Verified |

## Documentation and deliverables

| Deliverable | Location | Status | Note |
|---|---|---|---|
| README (setup, architecture, tool schemas, traces, decisions, limitations, improvements) | `README.md` | Verified | Commands checked against the code; `--live` trace run pending |
| 3-4 example traces incl. multi-tool and out-of-scope | `README.md`, `examples/sample_traces.jsonl` | Partial | Real graph output but scripted LLM + mock weather, labelled as such |
| Architecture diagram | `docs/architecture.md` (Mermaid) | Verified | Rendered successfully with mermaid-cli; matches module structure |
| Scalability discussion | `README.md` | Verified | Discussion only |
| Destination data pack ingested | `data/destinations/`, `scripts/ingest_data.py` | Verified | 20 chunks from 4 files; with offline embeddings only |
| Test cases | `tests/` | Verified | |
| GitHub repository | local git repo initialised; not pushed | Pending | Push needs your permission |
| Loom video (5-8 min) | n/a | **Pending** | Not recorded |
