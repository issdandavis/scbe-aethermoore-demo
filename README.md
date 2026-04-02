# SCBE-AETHERMOORE

**AI Governance. Mathematical Security. Built by Issac Davis.**

A 14-layer security pipeline using hyperbolic geometry to make AI attacks geometrically expensive. Patent pending (USPTO #63/961,403).

## Proven Results

| Experiment | Baseline Loss | Multi-view Loss | Improvement |
|-----------|--------------|----------------|-------------|
| Chat (Kaggle) | 2.2226 | 1.9121 | **14.0%** |
| Code (Colab T4) | 0.6339 | 0.5302 | **16.4%** |

Same model, same compute, same samples. Only difference: multi-view supervision across bytes, Sacred Tongues, and governance layers.

## What's Here

| Directory | What |
|-----------|------|
| `demos/` | 24 interactive browser demos (no install needed) |
| `research/` | Benchmark results, research pages |
| `src/polly_pump/` | The Pump -- inference-time orientation middleware |
| `products/benchmark-kit/` | $5 AI Governance Benchmark Kit (91 attacks, 10 classes) |
| `docs/specs/` | Architecture specs (Layer 12 formula, binary-first stack, etc.) |
| `tests/` | Adversarial test suite + pump tests |
| `artifacts/` | Experimental results (JSON) |

## The Pump Architecture

The Pump sits between your users and your model. It computes a tongue profile (6D domain analysis) and null pattern (what's absent) BEFORE the model generates. The model starts oriented, not drifting.

```
User query → SENSE (tongue profile) → LOCATE (aquifer match) → COMPOSE (pre-state) → Model → Response
```

**Canonical formula:** `H(d, pd) = 1 / (1 + φ * d_H + 2 * pd)`

## Six Sacred Tongues

| Tongue | Domain | Weight |
|--------|--------|--------|
| Kor'aelin (KO) | Control/Intent | 1.000 |
| Avali (AV) | Transport/Messaging | 1.618 |
| Runethic (RU) | Policy/Binding | 2.618 |
| Cassisivadan (CA) | Compute/Transforms | 4.236 |
| Umbroth (UM) | Security/Secrets | 6.854 |
| Draumric (DR) | Schema/Structure | 11.090 |

## Try It

- **Website:** [aethermoorgames.com](https://aethermoorgames.com)
- **npm:** `npm install scbe-aethermoore`
- **PyPI:** `pip install scbe-aethermoore`
- **HuggingFace:** [issdandavis/scbe-aethermoore-training-data](https://huggingface.co/datasets/issdandavis/scbe-aethermoore-training-data) (413K training pairs)
- **Book:** [The Six Tongues Protocol](https://www.amazon.com/dp/B0F28PHSPR) on Amazon

## Author

**Issac Davis** | Port Angeles, WA | ORCID: 0009-0002-3936-9369

The Six Sacred Tongues are original constructed languages. Cassisivadan encodes the creator's identity (Cassi = Issac reversed, Sivad = Davis reversed).

## License

MIT
