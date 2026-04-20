"""
risk_agent.py
Takes flagged clusters from Pattern Agent and assigns a risk score.

Multi-factor score (0 to 1):
    0.40 * time_factor      (how tight is the burst?)
    0.30 * ip_factor        (how many accounts on one IP?)
    0.30 * age_factor       (how young are the accounts?)

Output classification:
    score > 0.80  → HIGH risk, requires human review
    0.40 <= score <= 0.80  → MEDIUM risk, batch queue
    score < 0.40  → LOW risk, auto approve

This scoring is transparent on purpose. Every flagged case can be
explained to an auditor in one sentence.
"""


def score_cluster(cluster: dict) -> dict:
    """Assign a transparent multi-factor risk score to a flagged cluster."""

    # Time factor: shorter bursts are more suspicious.
    # 0 min burst → 1.0, 10 min burst → 0.0
    time_span = cluster["time_span_minutes"]
    time_factor = max(0.0, min(1.0, 1.0 - (time_span / 10.0)))

    # IP factor: more accounts sharing an IP is more suspicious.
    # 4 accounts → 0.5, 10+ accounts → 1.0
    acct_count = cluster["account_count"]
    ip_factor = min(1.0, acct_count / 10.0)

    # Age factor: younger accounts are more suspicious.
    # All young → 1.0, none young → 0.0
    age_factor = cluster["young_account_count"] / cluster["account_count"]

    score = 0.40 * time_factor + 0.30 * ip_factor + 0.30 * age_factor
    score = round(score, 3)

    if score > 0.80:
        classification = "HIGH"
        action = "Route to human auditor for immediate review"
    elif score >= 0.40:
        classification = "MEDIUM"
        action = "Add to daily batch review queue"
    else:
        classification = "LOW"
        action = "Auto-approve with audit trail"

    return {
        **cluster,
        "risk_score": score,
        "risk_factors": {
            "time_factor": round(time_factor, 3),
            "ip_factor": round(ip_factor, 3),
            "age_factor": round(age_factor, 3),
        },
        "classification": classification,
        "recommended_action": action,
    }


def score_all(clusters: list) -> list:
    return [score_cluster(c) for c in clusters]


if __name__ == "__main__":
    # Demo
    test_cluster = {
        "ip": "192.168.10.42",
        "sku": "SKU-PREMWATCH-7829",
        "seller": "SELLER-FRAUD-X",
        "account_count": 10,
        "young_account_count": 10,
        "time_span_minutes": 4.5,
        "total_amount": 5003.21,
    }
    scored = score_cluster(test_cluster)
    print(f"Score: {scored['risk_score']}")
    print(f"Classification: {scored['classification']}")
    print(f"Factors: {scored['risk_factors']}")
    print(f"Action: {scored['recommended_action']}")
