**Subject:** PRs ready for review — production hardening fixes (XS–M)

Hi team,

I've opened 7 PRs against ai-dynamo/dynamo covering the critical production issues we discussed:

**Fixes:**
- [#5860](https://github.com/ai-dynamo/dynamo/pull/5860) — NATS stream max_age 1h → 5min (memory leak under load)
- [#5861](https://github.com/ai-dynamo/dynamo/pull/5861) — NATS consumer inactive_threshold 1h → 2min (orphaned consumers on spot preemption)
- [#5862](https://github.com/ai-dynamo/dynamo/pull/5862) — Scheduler continues on watch error instead of exiting
- [#5863](https://github.com/ai-dynamo/dynamo/pull/5863) — KV subscriber continues on channel send failure instead of exiting
- [#5864](https://github.com/ai-dynamo/dynamo/pull/5864) — ZMQ production socket options (linger, keepalive, reconnect backoff)

**Performance:**
- [#5859](https://github.com/ai-dynamo/dynamo/pull/5859) — Demote per-request scheduler scoring logs to debug
- [#5865](https://github.com/ai-dynamo/dynamo/pull/5865) — Wrap sync tokenization in spawn_blocking (~50ms TTFT improvement)

All changes are tested locally with Qwen2.5-0.5B-Instruct on sglang, pass cargo check/clippy, and are intentionally small and self-contained.

Waiting for review on these — once merged we can move on to the larger items (routing by sequence length, SLA-based cancellation, Python→Rust JSON path).

Thanks
