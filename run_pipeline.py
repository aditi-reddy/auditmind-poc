"""
run_pipeline.py
Runs the full AuditMind POC end-to-end:

   Ingest  ->  Pattern Agent  ->  Risk Agent  ->  Alert Agent

Start with `python generate_mock_data.py` first to create the test dataset.
Then run this script.
"""

import os
import pattern_agent
import risk_agent
import alert_agent

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "transactions.csv")
OUT_DIR = os.path.join(HERE, "output")

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("AuditMind POC Pipeline")
    print("=" * 60)
    print()

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run `python generate_mock_data.py` first.")
        return

    # Step 1 + 2: Pattern Agent detects suspicious clusters
    print("[1/3] Pattern Agent: building graph and scanning for 刷单 patterns...")
    G, clusters, df = pattern_agent.run(CSV_PATH)

    if not clusters:
        print("No suspicious clusters detected. Exiting.")
        return

    # Step 3: Risk Agent scores each cluster
    print("[2/3] Risk Agent: scoring each flagged cluster...")
    scored = risk_agent.score_all(clusters)
    for s in scored:
        print(f"  Cluster on IP {s['ip']} → score {s['risk_score']} ({s['classification']})")
    print()

    # Step 4: Alert Agent writes the audit report
    print("[3/3] Alert Agent: generating audit report and gang graph...")
    report_path, graph_path = alert_agent.build_report(scored, total_txns=len(df), out_dir=OUT_DIR)
    print(f"  Report: {report_path}")
    print(f"  Graph:  {graph_path}")
    print()
    print("=" * 60)
    print("Pipeline complete. Open output/audit_report.md to see the result.")
    print("=" * 60)


if __name__ == "__main__":
    main()
