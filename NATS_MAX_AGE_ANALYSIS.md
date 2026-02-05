# Can we limit NATS stream max_age to 5 minutes?

Short answer: yes, but only if we also lower the snapshot threshold. Otherwise old routers catch up with stale data.

## Current defaults

- Stream max_age: 1 hour (nats.rs:514)
- Snapshot threshold: 1,000,000 messages (kv_router.rs:178)
- Consumer deliver policy: DeliverAll (implicit default)
- Consumer inactive threshold: 1 hour

## What happens on restart

When a router comes back, it does two things in order:
1. Downloads the last snapshot from NATS Object Store (a full dump of the radix tree)
2. Reconnects its durable JetStream consumer and starts receiving events

The snapshot gives historical state. The consumer gives new events. Together they should cover the full timeline.

## The gap problem

The snapshot is only taken when the stream hits 1M messages. At typical event rates that could be hours between snapshots. Meanwhile, with max_age=5min the stream only keeps the last 5 minutes of messages.

If a router was down for 20 minutes:
- Snapshot might be from 2 hours ago
- Stream only has the last 5 minutes
- Everything between the snapshot and the stream start is lost
- That's roughly 1h55m of KV cache events the router will never see

The router still works -- it just has a stale view of which workers hold which blocks. It routes suboptimally (more cache misses) until new events fill in the picture. No errors, no crashes, just worse routing for a while.

## When it's safe

Short restarts (<5min): the durable consumer picks up right where it left off, messages are still in the stream. No gap at all.

Frequent snapshots: if the threshold is low enough (say 10k instead of 1M), snapshots happen often and the gap between snapshot time and stream start stays small.

## Recommendation

Reducing max_age to 5min is fine if we also reduce router_snapshot_threshold from 1M to something like 10,000. This keeps snapshots fresh enough that the Object Store catch-up covers most of the window the stream no longer retains.

Without lowering the threshold, a router that was down for more than 5 minutes could start with a very stale view of the world.
