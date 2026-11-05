# Eval results — 2026-11-05T23:36:24

Models: small `qwen2.5:3b`, large `qwen2.5:7b`, embeddings `nomic-embed-text` · machine: Windows 11, 7.3 GB RAM, 12 CPUs · commit `c9dbf41` · wall time 50 min

| Category | Passed | Rate | Median time |
|---|---|---|---|
| analytics | 6/7 | 86% | 145 s |
| grounding | 3/3 | 100% | 72 s |
| memory | 1/2 | 50% | 157 s |
| planning | 8/8 | 100% | 16 s |
| research | 3/3 | 100% | 258 s |
| safety | 5/5 | 100% | 85 s |
| triage | 8/8 | 100% | 0 s |
| **all** | 34/36 | 94% | 59 s |

## Failures

- `a-early-starters`: contains all ['Ben', 'Dara', 'Gus'] — missing ['Dara']
- `m-preference`: recalled 'bullet' — memory: ''
