Makes sense, thanks for the detail — the Router 2 cold start scenario is the one I hadn't fully considered.

Reworked the PR: default stays at 1h, added docs for `DYN_NATS_STREAM_MAX_AGE` in `docs/design_docs/event_plane.md` so users can tune it when they know their snapshot frequency covers the gap.

Re local indexer - interested to try it on our sglang setup and report back, ping me or email at korolev.konstantin.v@gmail.com please.
