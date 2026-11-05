# Eval results — 2026-11-06T00:29:58

Models: small `qwen2.5:3b`, large `qwen2.5:7b`, embeddings `nomic-embed-text` · machine: Windows 11, 7.3 GB RAM, 12 CPUs · commit `0ee994e` · wall time 51 min

| Category | Passed | Rate | Median time |
|---|---|---|---|
| analytics | 6/7 | 86% | 136 s |
| grounding | 3/3 | 100% | 72 s |
| memory | 2/2 | 100% | 178 s |
| planning | 8/8 | 100% | 15 s |
| research | 3/3 | 100% | 212 s |
| safety | 5/5 | 100% | 82 s |
| triage | 8/8 | 100% | 0 s |
| **all** | 35/36 | 97% | 60 s |

## Failures

- `a-early-starters`: contains all ['Ben', 'Dara', 'Gus'] — missing ['Dara']
