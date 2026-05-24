# PersonaAgent Mock Evaluation Report

- Generated at: `2026-05-24T19:51:40.735879+00:00`
- JSON report: `mock_eval_report.json`

## Sample Size

| Dataset | Cases |
| --- | ---: |
| RAG | 2 |
| Memory | 2 |
| Style | 2 |
| Safety | 3 |
| LiteIM Integration | 4 |
| Total | 13 |

## Metrics

| Metric | Value |
| --- | ---: |
| RAG Hit@5 | 50.00% |
| Memory Hit@5 | 100.00% |
| Style Similarity | 75.00% |
| Verbatim Leakage Rate | 33.33% |
| Safety Violation Rate | 0.00% |
| Human Review Trigger Rate | 33.33% |
| Average latency | 63.500 ms |
| p95 latency | 81.000 ms |
| Token cost per reply | $0.00000000 |
| LiteIM integration success rate | 100.00% |

## A/B Variants

| Variant | Cases | Avg latency | p95 latency | Cost/reply | Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| No RAG | 1 | 42.000 ms | 42.000 ms | $0.00000000 | 100.00% |
| Knowledge only | 1 | 58.000 ms | 58.000 ms | $0.00000000 | 100.00% |
| Knowledge + Memory | 1 | 73.000 ms | 73.000 ms | $0.00000000 | 100.00% |
| Knowledge + Memory + Style | 1 | 81.000 ms | 81.000 ms | $0.00000000 | 100.00% |
