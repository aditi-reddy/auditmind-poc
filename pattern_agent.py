"""
pattern_agent.py
Builds a transaction network graph and detects suspicious coordinated patterns.

Detection passes:
  Pass A - Shared-IP burst:
      N or more young accounts share one IP, buy the same SKU,
      within a short time window. Classic 刷单 signature.

  Pass B - Shared-device burst (VPN rotation case):
      N or more young accounts share one physical device but use
      rotating IPs (typical of VPN-masked fraud farms), buy the
      same SKU within a short window.

Both passes feed into the same risk scoring pipeline. In the production
build these rule-based passes are replaced with an RGCN model trained
on labeled fraud cases. Keeping them rule-based in the POC means every
flagged case can be explained to an auditor in one sentence.
"""

import pandas as pd
import networkx as nx
from datetime import datetime
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


def _cluster_record(group, group_sorted, shared_key, shared_value, sku, pattern_type):
    """Build a cluster dict from a pandas group."""
    time_span = (group_sorted["timestamp"].iloc[-1] - group_sorted["timestamp"].iloc[0]).total_seconds() / 60
    young_count = (group_sorted["account_age_days"] <= 7).sum()

    rec = {
        "pattern_type": pattern_type,
        "shared_key": shared_key,
        "shared_value": shared_value,
        "sku": sku,
        "account_count": len(group),
        "young_account_count": int(young_count),
        "time_span_minutes": round(time_span, 2),
        "accounts": list(group["account_id"].unique()),
        "devices": list(group["device_id"].unique()),
        "ips": list(group["ip_address"].unique()),
        "seller": group["seller_id"].iloc[0],
        "total_amount": round(group["amount"].sum(), 2),
        "transactions": group["transaction_id"].tolist(),
        "first_seen": group_sorted["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "last_seen": group_sorted["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Back-compat field for Alert Agent
    rec["ip"] = shared_value if shared_key == "ip_address" else group["ip_address"].iloc[0]
    return rec


def detect_shared_ip_burst(df, min_accounts=4, time_window_minutes=10, young_days=7):
    """Pass A: multiple young accounts sharing one IP, same SKU, tight window."""
    clusters = []
    for (ip, sku), group in df.groupby(["ip_address", "sku"]):
        if len(group) < min_accounts:
            continue
        group_sorted = group.sort_values("timestamp")
        time_span = (group_sorted["timestamp"].iloc[-1] - group_sorted["timestamp"].iloc[0]).total_seconds() / 60
        if time_span > time_window_minutes:
            continue
        # Require at least SOME young accounts (softer for Pattern 3 medium case)
        young_count = (group_sorted["account_age_days"] <= young_days * 2).sum()
        if young_count < min_accounts:
            continue
        clusters.append(_cluster_record(group, group_sorted, "ip_address", ip, sku, "shared_ip_burst"))
    return clusters


def detect_shared_device_burst(df, min_accounts=4, time_window_minutes=10, young_days=7):
    """Pass B: multiple young accounts sharing one device, same SKU, tight window."""
    clusters = []
    for (device, sku), group in df.groupby(["device_id", "sku"]):
        if len(group) < min_accounts:
            continue
        group_sorted = group.sort_values("timestamp")
        time_span = (group_sorted["timestamp"].iloc[-1] - group_sorted["timestamp"].iloc[0]).total_seconds() / 60
        if time_span > time_window_minutes:
            continue
        young_count = (group_sorted["account_age_days"] <= young_days * 2).sum()
        if young_count < min_accounts:
            continue
        # Extra signal: IPs rotated? (device shared but IPs differ)
        unique_ips = group["ip_address"].nunique()
        rec = _cluster_record(group, group_sorted, "device_id", device, sku, "shared_device_burst")
        rec["unique_ip_count"] = int(unique_ips)
        clusters.append(rec)
    return clusters


def dedupe_clusters(clusters):
    """
    Remove clusters that are subsets of larger ones (same gang seen from two angles).
    When a shared-device and shared-IP pass flag overlapping account sets, the
    larger cluster wins because it carries more evidence.
    """
    # Sort largest first so we keep biggest clusters
    sorted_clusters = sorted(clusters, key=lambda c: -len(c["accounts"]))
    kept = []
    for c in sorted_clusters:
        c_accts = set(c["accounts"])
        is_subset_of_kept = False
        for k in kept:
            k_accts = set(k["accounts"])
            # If this cluster's accounts are mostly inside a kept one, skip it
            overlap = len(c_accts & k_accts)
            if overlap >= 0.6 * len(c_accts):
                is_subset_of_kept = True
                break
        if not is_subset_of_kept:
            kept.append(c)
    return kept


def run(csv_path: str):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    G = build_graph(df)
    pass_a = detect_shared_ip_burst(df)
    pass_b = detect_shared_device_burst(df)
    clusters = dedupe_clusters(pass_a + pass_b)
    # Sort: largest / most suspicious first
    clusters.sort(key=lambda c: (-c["account_count"], c["time_span_minutes"]))

    print(f"Loaded {len(df)} transactions.")
    print(f"Built network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Pass A (shared-IP burst) fired:     {len(pass_a)} cluster(s)")
    print(f"Pass B (shared-device burst) fired: {len(pass_b)} cluster(s)")
    print(f"Unique clusters after dedupe:       {len(clusters)}")
    print()

    for i, c in enumerate(clusters, 1):
        print(f"--- Cluster #{i} [{c['pattern_type']}] ---")
        if c["pattern_type"] == "shared_ip_burst":
            print(f"  Shared IP:       {c['shared_value']}")
        else:
            print(f"  Shared device:   {c['shared_value']}")
            print(f"  IPs used:        {c.get('unique_ip_count', 1)} (VPN rotation suspected)")
        print(f"  SKU:             {c['sku']}")
        print(f"  Seller:          {c['seller']}")
        print(f"  Accounts:        {c['account_count']} ({c['young_account_count']} young)")
        print(f"  Time window:     {c['time_span_minutes']} minutes")
        print(f"  Total amount:    ${c['total_amount']:.2f}")
        print()

    return G, clusters, df


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")
    run(csv_path)
