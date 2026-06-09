# 🔍 AuditMind POC

### Automated Multi-Agent E-Commerce Fraud & Gang Detection Pipeline
[![挑戰赛](https://img.shields.io/badge/Competition-2026%20Deloitte%20Digital%20Elite%20Challenge-blue.svg?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=flat-square)](#)
[![Graph Framework](https://img.shields.io/badge/Framework-NetworkX-orange.svg?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Complete-green.svg?style=flat-square)](#)

---

## 🎯 Project Overview

**AuditMind POC** is a fully runnable proof-of-concept demonstrating the multi-agent fraud detection pipeline proposed for the **2026 Deloitte Digital Elite Challenge (Team I)**. 

The system automates the identification of coordinated **"刷单" (fake order/sales brushing) gangs** on e-commerce platforms. It processes transaction feeds, builds behavioral network graphs, identifies anomalies, scores risk profile weights, and compiles markdown reports with graphical visualizations of the illicit network.

---

## 🤖 Multi-Agent Pipeline Architecture

AuditMind orchestrates three specialized agents, each executing a distinct phase of the detection workflow:

```
[ Transaction Stream (JSON/CSV) ]
              │
              ▼
    ╔───────────────────╗
    │   Pattern Agent   │  <─── Builds NetworkX Graph & scans for 
    ╚─────────┬─────────╝       multi-dimensional coordinated signatures
              │ (Flagged Clusters)
              ▼
    ╔───────────────────╗
    │    Risk Agent     │  <─── Computes composite risk weights (0.0 - 1.0)
    ╚─────────┬─────────╝       based on velocity, age, & volume metrics
              │ (Scored Gangs)
              ▼
    ╔───────────────────╗
    │    Alert Agent    │  <─── Renders visualization graph (.PNG)
    ╚───────────────────╝       & generates Markdown Audit Reports
```

1. **Pattern Agent**: Constructs a heterogeneous network graph where nodes represent **Accounts, IPs, Devices, SKUs, and Sellers**. It runs rule-based pattern mining (a proxy for the full system's Relational Graph Convolutional Network - RGCN) to isolate clusters where multiple accounts share connection parameters within tight time windows.
2. **Risk Agent**: Evaluates the isolated clusters using a multi-factor risk model scoring velocity, account age profiles, seller exposure, and transaction value.
3. **Alert Agent**: Generates actionable intelligence including an executive audit report and visual graph networks mapping nodes and transaction edges.

---

## ⚙️ The Detection & Scoring Logic

### 1. Pattern Detection Rules
A gang signature is flagged by the **Pattern Agent** when:
- **$\ge$ 4 buyer accounts** share a single IP address or Device fingerprint.
- All flagged accounts purchase the **same SKU** (product).
- The transactions occur within a tight **10-minute window**.
- A majority ($\ge$ 60%) of the accounts are **young** (created < 7 days prior).

### 2. Risk Scoring Formula
The **Risk Agent** calculates a composite risk index ($R \in [0, 1]$) based on four weighted indicators:
$$R = w_1 \cdot V_{time} + w_2 \cdot A_{age} + w_3 \cdot S_{size} + w_4 \cdot T_{value}$$

Where:
* **Time Velocity ($V_{time}$)**: Inverse log of the time window size.
* **Account Age ($A_{age}$)**: Ratio of newly created accounts to total accounts.
* **Gang Size ($S_{size}$)**: Normalized scale of unique participant nodes.
* **Total Transaction Value ($T_{value}$)**: Log-normalized monetary value.

Clusters scoring $\ge 0.70$ are flagged as **HIGH RISK** and routed for immediate human auditing.

---

## 🛠️ Technical Stack

- **Core Engine**: Python 3.11+
- **Network & Graph Modeling**: `networkx`
- **Visualization**: `matplotlib`, `seaborn`
- **Data Structuring**: `pandas`, `numpy`

---

## 🚀 Getting Started & Reproducibility

To generate mock data, run the detection pipeline, and verify the outputs:

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/aditi-reddy/auditmind-poc.git
cd auditmind-poc
pip install -r requirements.txt
```

### 2. Execution
Run the synthetic data generator and execute the agent pipeline:
```bash
# Generate 50 transaction records containing a hidden coordination gang
python generate_mock_data.py

# Execute Pattern, Risk, and Alert agent tasks
python run_pipeline.py
```

### 3. Review Outputs
The pipeline outputs results directly to the `output/` directory:
- `output/audit_report.md` — A clear, structured Markdown report outlining the flagged gang, its member nodes, and risk calculations.
- `output/gang_graph.png` — A matplotlib-rendered network graph visualizing nodes (Accounts, IPs, SKUs) and connection edges.
