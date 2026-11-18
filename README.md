# Avalon Harness

The multi-agent harness [Avalon-M1-Mini](https://huggingface.co/GeneralReasoning/Avalon-M1-Mini)
was trained in. It implements **AgentOrg**, the organisation protocol behind
the model. A session is one organisation: a CEO agent reads the task, hires
specialist agents (up to an organisation budget), and the team works through
shared channels, direct messages, a shared archive and a shared workspace, with
the CEO submitting the final result. Every agent is a copy of the same model
with its own context.

The protocol is specified in [`SPEC.md`](SPEC.md). The model's prompts call
the session an AgentOrg, and the Python package is `agentorg`.

> **Research use only.** Avalon-M1-Mini is a research model for multi-agent
> systems and multi-agent safety, not intended for production use. Agents can
> run shell commands and act on their environment: run the harness and its
> environments in isolated sandboxes.

## Install

```bash
uv sync
```

This installs two commands: `avalon` runs tasks and `avalon-inspect` reads the
rollouts they write. Run them with `uv run`, or activate `.venv`.

## Serve the model

The harness talks to any OpenAI-compatible server with tool calling. For
Avalon-M1-Mini, with SGLang:

```bash
python -m sglang.launch_server \
  --model-path GeneralReasoning/Avalon-M1-Mini \
  --trust-remote-code --tp 8 --context-length 131072 \
  --reasoning-parser glm45 --tool-call-parser qwen3_coder
```

Or with vLLM:

```bash
vllm serve GeneralReasoning/Avalon-M1-Mini \
  --trust-remote-code --tensor-parallel-size 8 --max-model-len 131072 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser glm45 --port 30000
```

Both were tested with SGLang 0.5.23 and vLLM 0.34.0 on 8 H200 GPUs.
See the [model card](https://huggingface.co/GeneralReasoning/Avalon-M1-Mini) for details.

## Run tasks

Tasks come from [OpenReward](https://openreward.ai) environments:

```bash
export OPENREWARD_API_KEY=...

uv run avalon \
    --env GeneralReasoning/Chernobyl \
    --limit 1 \
    --budget 4
```

`avalon` sends model requests to the server started above, at
`http://localhost:30000/v1`. If the model is served elsewhere, pass
`--base-url` (and `--api-key` if that server requires a key), or set
`AVALON_BASE_URL` / `AVALON_API_KEY`. Any server that speaks the OpenAI Chat
Completions API with tool calling works.

`--budget N` is the maximum number of specialists alive at once (default 2;
0 to 8 in training; 0 runs a single agent). Firing a specialist frees its
slot. `--split` defaults to `train`; use `--split test` for held-out tasks
where the environment has a test split. With one rollout the CLI streams a live trace of the
organisation; `--concurrency` runs several tasks in parallel. Environments
whose graders need their own credentials take `--secret KEY[=ENV_VAR]`
(for example `--secret openai_api_key`); the value is read from the
environment, never from the command line.

### Training defaults

The CLI defaults match how Avalon-M1-Mini was trained:

| Setting | Value |
| --- | --- |
| Mode | `org-plan` (the CEO drafts an organisation before acting) |
| Temperature | 1.0, no top-p |
| Context window | 131,072 tokens |
| Full compaction | at 114,688 tokens, keeping up to 32,768 tokens of recent turns verbatim, 12,000-token summaries |
| Microcompaction | off |
| Date in system prompt | none (`--include-date` adds one) |

For another model, pass `--model`, point `--base-url` at its server, and
give `--context-window-tokens` so compaction thresholds are derived from that
model's window.

### Rewards

The reported score (`reward` in the CLI output and `RolloutResult.reward`) is
the environment's reward. The harness also stamps a no-op `shaping_reward`
into tool results the model sees (for shell calls that do nothing, such as
`exit 0`), as it did in training, but never adds it to the score.
Avalon-M1-Mini's training reward was the environment reward only, shared by
the whole organisation.

### Output

Each rollout writes a session directory under
`rollouts/<env>__<split>/<run-id>/task-NNNN/`: transcripts per agent, channel
logs, the archive, an event log, and a `trajectory.json` summary. To look
inside one:

```bash
uv run avalon-inspect rollouts/<env>__<split>/<run-id>/task-0000
uv run avalon-inspect <session-dir> --transcript CEO
uv run avalon-inspect <session-dir> --channel announcements
```

## Library use

[`examples/programmatic_usage.py`](examples/programmatic_usage.py) builds a
`Harness`, binds a session to an OpenReward task and runs it to completion.
The library defaults are `mode="default"` and generic compaction thresholds;
to reproduce training, a `SessionConfig` needs `mode="org-plan"` and
`compaction=avalon_compaction_config()` (from `agentorg.compaction`).
Temperature 1.0 is already the default.

## Tests

```bash
uv run --extra dev pytest
```

## Paper and citation

The training recipe and analysis are described in the
[Avalon-M1 release](https://www.gr.inc/releases/avalon-m1).

```bibtex
@misc{avalonm1,
  title  = {Avalon-M1: training agent organisations with reinforcement learning from collective rewards},
  author = {General Reasoning},
  year   = {2026},
  url    = {https://www.gr.inc/releases/avalon-m1}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
