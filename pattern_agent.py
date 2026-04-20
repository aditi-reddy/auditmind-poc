"""
pattern_agent.py
Builds a transaction network graph and detects suspicious coordinated patterns.

How it works:
  1. Loads the transactions CSV
  2. Builds a graph: nodes = accounts, IPs, devices, SKUs, sellers
                     edges = "uses IP", "uses device", "buys SKU", "pays seller"
  3. Scans the graph for clusters where multiple young accounts share
     the same IP + are buying the same SKU within a short time window.
  4. Returns the flagged clusters with all supporting evidence.

This is the Pattern Agent, simplified but real. For production, the scan
would be replaced with an RGCN model trained on labeled fraud cases.
"""

import pandas as pd
import networkx as nx
from datetime import datetime
from collections import defaultdict
import os


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a heterogeneous transaction graph."""
    G = nx.Graph()

    for _, row in df.iterrows():
        acct = f"A:{row['account_id']}"
        ip = f"IP:{row['ip_address']}"
        dev = f"D:{row['device_id']}"
        sku = f"S:{row['sku']}"
        seller = f"SL:{row['seller_id']}"

        G.add_node(acct, kind="account", age=row["account_age_days"])
        G.add_node(ip, kind="ip")
        G.add_node(dev, kind="device")
        G.add_node(sku, kind="sku")
        G.add_node(seller, kind="seller")

        G.add_edge(acct, ip, relation="uses_ip")
        G.add_edge(acct, dev, relation="uses_device")
        G.add_edge(acct, sku, relation="bought", ts=row["timestamp"], amount=row["amount"])
        G.add_edge(sku, seller, relation="sold_by")

    return G


def detect_shared_ip_burst(
    df: pd.DataFrame,
    min_accounts: int = 4,
    time_window_minutes: int = 10,
    young_account_threshold_days: int = 7,
):
    """
    Core 刷单 detector.

    Flag: N or more YOUNG accounts share the same IP AND buy the same SKU
    within a short time window.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    clusters = []

    # Group by (ip, sku)
    grouped = df.groupby(["ip_address", "sku"])
    for (ip, sku), group in grouped:
        if len(group) < min_accounts:
            continue

        # Time window filter: are they clustered in time?
        group_sorted = group.sort_values("timestamp")
        time_span = (group_sorted["timestamp"].iloc[-1] - group_sorted["timestamp"].iloc[0]).total_seconds() / 60
        if time_span > time_window_minutes:
            continue

        # Age filter: are most accounts young?
        young_count = (group_sorted["account_age_days"] <= young_account_threshold_days).sum()
        if young_count < min_accounts:
            continue

        clusters.append({
            "ip": ip,
            "sku": sku,
            "account_count": len(group),
            "young_account_count": int(young_count),
            "time_span_minutes": round(time_span, 2),
            "accounts": list(group["account_id"].unique()),
            "devices": list(group["device_id"].unique()),
            "seller": group["seller_id"].iloc[0],
            "total_amount": round(group["amount"].sum(), 2),
            "transactions": group["transaction_id"].tolist(),
            "first_seen": group_sorted["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "last_seen": group_sorted["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        })

    return clusters


def run(csv_path: str):
    df = pd.read_csv(csv_path)
    G = build_graph(df)
    clusters = detect_shared_ip_burst(df)

    print(f"Loaded {len(df)} transactions.")
    print(f"Built network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Detected {len(clusters)} suspicious cluster(s).\n")

    for i, c in enumerate(clusters, 1):
        print(f"--- Suspicious cluster #{i} ---")
        print(f"  IP:              {c['ip']}")
        print(f"  SKU:             {c['sku']}")
        print(f"  Seller:          {c['seller']}")
        print(f"  Accounts:        {c['account_count']} ({c['young_account_count']} young)")
        print(f"  Time window:     {c['time_span_minutes']} minutes")
        print(f"  Total amount:    ${c['total_amount']:.2f}")
        print(f"  First seen:      {c['first_seen']}")
        print()

    return G, clusters, df


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")
    run(csv_path)
