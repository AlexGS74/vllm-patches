# vllm-patches — sm120-nvfp4

Cherry-picked patches on top of vLLM `main` for running GLM-4.7 NVFP4 on
SM120 (RTX PRO 6000 Blackwell) hardware.

Upstream: https://github.com/vllm-project/vllm

---

## Applied patches

### 1. [Anthropic API] Fix 6 protocol compliance bugs in Messages endpoint
- **Upstream PR:** [#34053](https://github.com/vllm-project/vllm/pull/34053) (open as of 2026-02-28)
- **Author:** SilviuSavu
- **Cherry-picked:** `a76a77d9f`, `275b8d1ef`
- **Date applied:** 2026-02-28

Fixes six bugs in vLLM's `/v1/messages` Anthropic-compatible endpoint that
break Claude Code compatibility:

| # | Bug | Fix |
|---|-----|-----|
| 1 | Response IDs use `chatcmpl-*` format | Changed to `msg_` prefix |
| 2 | Tool call IDs use `call_` + timestamp | Changed to `toolu_` prefix |
| 3 | Tool result content (list-of-blocks) converted to Python `repr` string | Proper extraction via `_extract_tool_result_text` |
| 4 | Spurious empty text blocks in non-streaming tool responses | Skip empty content blocks |
| 5 | First chunk of streaming tool call arguments silently dropped | Emit as `input_json_delta` |
| 6 | Unused `time` import | Removed |

**Conflict resolution notes:**
- C1: Kept `import uuid` (used by thinking block signature), dropped `import time`
- C2: Took PR — `block.id` and `_extract_tool_result_text` are correct
- C3: Merged — kept HEAD's reasoning/thinking block support + PR's skip-empty-text guard
- C4: Kept HEAD's `reasoning_delta` streaming; PR's first-chunk args fix was already
  present in `main` at the tool_call streaming block (lines ~628+)

---

## Already in main (no cherry-pick needed)

### GLM-4.5 / GLM-4.7 enable_thinking fix
- **Upstream PR:** [#31788](https://github.com/vllm-project/vllm/pull/31788) (merged 2026-01-06)
- Fixes `enable_thinking: false` being silently ignored for GLM-4.7, causing
  answers to land in `reasoning_content` with `content: null`
- **Status:** In `main` as of the fork date — already present on this branch

---

## Pending / candidates

| PR | Description | Status |
|----|-------------|--------|
| [#32997](https://github.com/vllm-project/vllm/pull/32997) | Prevent reasoning_content leak on tool_calls finish | Open |
| SM120 FlashInfer FP4 MoE | Fix FlashInfer broken on SM120 for FP4 MoE (currently using TRITON_ATTN workaround) | Investigating |
