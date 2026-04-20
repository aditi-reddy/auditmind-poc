# AuditMind POC

**A working proof-of-concept for the 2026 Deloitte Digital Elite Challenge (Team I).**

This repository is the runnable companion to my AuditMind proposal. It is a minimal but real implementation of the multi-agent fraud detection pipeline I described in the submission document: Pattern Agent → Risk Agent → Alert Agent, working on a mock e-commerce dataset with a planted 刷单 (fake order) gang.

It is not the full AuditMind system. It is proof that I can actually build one.

---

## What it does

1. Generates 50 mock transactions (40 normal + 10 coordinated fraud)
2. **Pattern Agent** builds a network graph and scans for 刷单 signatures
3. **Risk Agent** scores each flagged cluster using a transparent multi-factor formula
4. **Alert Agent** writes a Markdown audit report and renders the gang as a PNG

All in under 300 lines of Python. Everything is reproducible.

---

## How to run

```bash
pip install -r requirements.txt
python generate_mock_data.py
python run_pipeline.py
```

Outputs:
- `output/audit_report.md` — human-readable audit report
- `output/gang_graph.png` — visual of the detected fraud network

---

## Example output on the mock dataset

```
Loaded 50 transactions.
Built network graph with 203 nodes and 191 edges.
Detected 1 suspicious cluster(s).

Cluster on IP 192.168.10.42 → score 0.865 (HIGH)
  Accounts:        10 (10 young)
  Time window:     3.37 minutes
  SKU:             SKU-PREMWATCH-7829
  Seller:          SELLER-FRAUD-X
  Total amount:    $5043.95
```

The system correctly identifies the planted 10-account gang and assigns it HIGH risk, recommending immediate human auditor review.

---

## The detection logic (Pattern Agent)

The graph has four node types and four edge types:

| Node | Meaning |
|---|---|
| Account | A buyer account |
| IP | An IP address |
| Device | A device fingerprint |
| SKU | A product being purchased |
| Seller | The seller receiving the revenue |

A cluster is flagged when:

- **4 or more accounts** share the same IP address
- All buy the **same SKU**
- Within a **10-minute window**
- **Most accounts are less than 7 days old**

This rule captures the core coordinated 刷单 signature. In the full AuditMind system, this rule-based scan is replaced with an RGCN (Relational Graph Convolutional Network) trained on labeled fraud cases. For the POC, a transparent rule is more useful because every flagged case can be explained to an auditor in one sentence.

---

## The scoring logic (Risk Agent)

Score is a weighted sum of three factors, each normalized to [0, 1]:

```
score = 0.40 * time_factor   (tighter burst = higher risk)
      + 0.30 * ip_factor     (more accounts on one IP = higher risk)
      + 0.30 * age_factor    (younger accounts = higher risk)
```

Thresholds:

| Score | Class | Action |
|---|---|---|
| > 0.80 | HIGH | Human auditor review required |
| 0.40 to 0.80 | MEDIUM | Daily batch review queue |
| < 0.40 | LOW | Auto-approve with audit trail |

This is the Human-in-the-Loop (HITL) pattern from the proposal. No account action is taken without a human confirming.

---

## How this connects to the full AuditMind proposal

This POC is the thin end of the wedge. The pieces it proves are:

- **Ingest**: I can load, clean, and structure transaction data (covered by my existing personal-finance-pipeline)
- **Pattern**: I can build a graph and detect coordinated patterns across accounts
- **Risk**: I can combine multiple signals into a single explainable score
- **Alert**: I can generate audit-quality output that humans actually use
- **HITL**: The routing logic separates what needs a human from what does not

The pieces that are **not** in this POC, which I would build during the finals phase:

- RGCN model trained on real labeled data (replacing the rule-based scan)
- LLM integration for natural-language audit narratives in Alert Agent
- Three-layer memory system (working, episodic, semantic) with real persistence
- DingTalk/Slack webhook delivery
- On-premise deployment for Chinese enterprise clients

---

## Repository layout

```
auditmind-poc/
├── README.md                  this file
├── requirements.txt           pandas, networkx, matplotlib, numpy
├── generate_mock_data.py      creates 50-row test dataset
├── pattern_agent.py           graph construction + detection
├── risk_agent.py              multi-factor scoring
├── alert_agent.py             report generation + visualization
├── run_pipeline.py            end-to-end runner
├── data/
│   └── transactions.csv       generated dataset
└── output/
    ├── audit_report.md        generated report
    └── gang_graph.png         generated network visualization
```

---

## Author

**Aditi Malla**
Central Michigan University, MS Information Systems
Submitted to the 2026 Deloitte Digital Elite Challenge, Team I (Mentor: Leo Ma)

- GitHub: [github.com/aditi-reddy](https://github.com/aditi-reddy)
- Existing data pipeline (foundation): [personal-finance-pipeline](https://github.com/aditi-reddy/personal-finance-pipeline)
- LinkedIn: [linkedin.com/in/aditi-reddy-malla-275a70222](https://www.linkedin.com/in/aditi-reddy-malla-275a70222/)

---

## License

This project is submitted for the Deloitte Digital Elite Challenge 2026. All code is original work.
