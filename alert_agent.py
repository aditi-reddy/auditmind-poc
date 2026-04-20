"""
alert_agent.py
Turns scored clusters into a human-readable Markdown audit report
AND renders the fraud gang as a PNG graph for the report.

In production, this would push to DingTalk / Slack / email.
For the POC, it writes:
  output/audit_report.md
  output/gang_graph.png
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime


def render_gang_graph(cluster: dict, out_path: str):
    """Draw the gang network as a PNG."""
    G = nx.Graph()

    ip_node = f"IP\n{cluster['ip']}"
    sku_node = f"SKU\n{cluster['sku']}"
    seller_node = f"Seller\n{cluster['seller']}"

    G.add_node(ip_node, kind="ip")
    G.add_node(sku_node, kind="sku")
    G.add_node(seller_node, kind="seller")

    for acct in cluster["accounts"]:
        acct_node = acct
        G.add_node(acct_node, kind="account")
        G.add_edge(ip_node, acct_node)
        G.add_edge(acct_node, sku_node)

    G.add_edge(sku_node, seller_node)

    color_map = {
        "account": "#e74c3c",
        "ip": "#f39c12",
        "sku": "#9b59b6",
        "seller": "#2c3e50",
    }
    sizes = {"account": 1200, "ip": 2200, "sku": 2000, "seller": 2200}

    node_colors = [color_map[G.nodes[n]["kind"]] for n in G.nodes()]
    node_sizes = [sizes[G.nodes[n]["kind"]] for n in G.nodes()]

    pos = nx.spring_layout(G, seed=7, k=1.2)

    plt.figure(figsize=(11, 7))
    nx.draw_networkx_edges(G, pos, edge_color="#999", width=1.2)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=7.5, font_color="white", font_weight="bold")
    plt.title("AuditMind Pattern Agent: flagged fraud gang network", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def build_report(scored_clusters: list, total_txns: int, out_dir: str):
    report_path = os.path.join(out_dir, "audit_report.md")
    graph_path = os.path.join(out_dir, "gang_graph.png")

    # Render graph for the first high-risk cluster
    if scored_clusters:
        render_gang_graph(scored_clusters[0], graph_path)

    lines = []
    lines.append("# AuditMind Fraud Detection Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Transactions analyzed:** {total_txns}")
    lines.append(f"**Suspicious clusters flagged:** {len(scored_clusters)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not scored_clusters:
        lines.append("No suspicious activity detected in this batch.")
    else:
        for i, c in enumerate(scored_clusters, 1):
            lines.append(f"## Cluster {i}: {c['classification']} RISK (score {c['risk_score']})")
            lines.append("")
            lines.append(f"**Recommended action:** {c['recommended_action']}")
            lines.append("")
            lines.append("### Why this was flagged")
            lines.append("")
            lines.append(f"- **{c['account_count']} accounts** all used the same IP address `{c['ip']}`")
            lines.append(f"- **{c['young_account_count']} of {c['account_count']}** accounts were created in the last 7 days")
            lines.append(f"- All **{c['account_count']} accounts** bought the same SKU `{c['sku']}`")
            lines.append(f"- All purchases happened within **{c['time_span_minutes']} minutes**")
            lines.append(f"- Revenue concentrated on a single seller: `{c['seller']}`")
            lines.append(f"- Total amount: **${c['total_amount']:.2f}**")
            lines.append("")
            lines.append("### Risk score breakdown")
            lines.append("")
            lines.append("| Factor | Value | Weight | Contribution |")
            lines.append("|---|---|---|---|")
            tf = c["risk_factors"]["time_factor"]
            ipf = c["risk_factors"]["ip_factor"]
            af = c["risk_factors"]["age_factor"]
            lines.append(f"| Time burst tightness | {tf} | 0.40 | {round(tf*0.40, 3)} |")
            lines.append(f"| Shared-IP density | {ipf} | 0.30 | {round(ipf*0.30, 3)} |")
            lines.append(f"| Account youth | {af} | 0.30 | {round(af*0.30, 3)} |")
            lines.append(f"| **Total** |  |  | **{c['risk_score']}** |")
            lines.append("")
            lines.append("### Involved accounts")
            lines.append("")
            for a in c["accounts"]:
                lines.append(f"- `{a}`")
            lines.append("")
            lines.append("### Network graph")
            lines.append("")
            lines.append(f"![Gang graph](gang_graph.png)")
            lines.append("")
            lines.append("### Transaction IDs")
            lines.append("")
            lines.append("```")
            for t in c["transactions"]:
                lines.append(t)
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    lines.append("*Generated by AuditMind Pattern Agent + Risk Agent + Alert Agent*")
    lines.append("")
    lines.append("*Human auditor confirmation required before any account action.*")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path, graph_path
