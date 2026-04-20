"""
generate_mock_data.py
Generates a realistic e-commerce transaction dataset with THREE distinct planted 刷单 patterns.

Output: data/transactions.csv (1,523 rows total)

Planted fraud patterns (based on real Chinese e-commerce case studies):

  Pattern 1 - IP-sharing gang (HIGH risk expected)
    10 very young accounts, all on one IP, same SKU,
    tight time burst. Classic 刷单 operation.

  Pattern 2 - Device-rotation gang (HIGH risk expected)
    8 accounts using different IPs (VPN rotation) but the
    same physical device, synchronized on a different SKU.

  Pattern 3 - Mild coordination (MEDIUM risk expected)
    5 accounts, shared IP, same SKU, but older accounts and
    a longer time window. Tests that the scorer discriminates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")

# ---------- normal background traffic (1500 rows) ----------

normal_rows = []
base_time = datetime(2026, 3, 1, 0, 0, 0)

categories = ["BOOK", "SHOE", "PHONE", "LAMP", "BAG", "MUG", "TOY", "HAT", "PEN",
              "CUP", "SHIRT", "WATCH", "SOCKS", "DESK", "CHAIR", "TABLE"]

for i in range(1500):
    account_id = f"ACC-{1000 + i:05d}"
    # Established users: 30 to 900 days old
    account_age_days = random.randint(30, 900)
    ip = f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
    device = f"DEV-{random.randint(100000, 999999)}"
    sku = f"SKU-{random.choice(categories)}-{random.randint(100,999)}"
    seller = f"SELLER-{random.randint(1, 150):03d}"
    timestamp = base_time + timedelta(
        days=random.randint(0, 29),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    amount = round(random.uniform(15, 400), 2)
    normal_rows.append({
        "transaction_id": f"TXN-{10000 + i:06d}",
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "account_id": account_id,
        "account_age_days": account_age_days,
        "ip_address": ip,
        "device_id": device,
        "sku": sku,
        "seller_id": seller,
        "amount": amount,
    })

# ---------- Pattern 1: IP-sharing gang (HIGH risk) ----------
# 10 young accounts all hitting one IP, same SKU, tight burst.

p1_rows = []
p1_ip = "192.168.10.42"
p1_device_shared = "DEV-FRAUD-001"
p1_sku = "SKU-PREMWATCH-7829"
p1_seller = "SELLER-FRAUD-X"
p1_burst_start = datetime(2026, 3, 28, 23, 47, 0)

for i in range(10):
    account_id = f"ACC-GANG1-{i:02d}"
    account_age_days = random.randint(2, 5)
    ip = p1_ip
    device = p1_device_shared if i < 5 else f"DEV-FRAUD-{i:03d}"
    timestamp = p1_burst_start + timedelta(seconds=random.randint(0, 270))
    amount = round(random.uniform(480, 520), 2)
    p1_rows.append({
        "transaction_id": f"TXN-{200000 + i:06d}",
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "account_id": account_id,
        "account_age_days": account_age_days,
        "ip_address": ip,
        "device_id": device,
        "sku": p1_sku,
        "seller_id": p1_seller,
        "amount": amount,
    })

# ---------- Pattern 2: Device-rotation gang (HIGH risk) ----------
# 8 accounts using rotating IPs (VPN) but all on the same physical device.
# Same SKU, synchronized burst. Harder to catch with IP-only rules.

p2_rows = []
p2_device = "DEV-MULE-777"
p2_sku = "SKU-LUXEBAG-4421"
p2_seller = "SELLER-MULE-Y"
p2_burst_start = datetime(2026, 3, 24, 14, 22, 0)

for i in range(8):
    account_id = f"ACC-GANG2-{i:02d}"
    account_age_days = random.randint(3, 6)
    # Different IPs for each account to simulate VPN rotation
    ip = f"45.{random.randint(50,200)}.{random.randint(1,254)}.{random.randint(1,254)}"
    device = p2_device
    timestamp = p2_burst_start + timedelta(seconds=random.randint(0, 300))
    amount = round(random.uniform(890, 920), 2)
    p2_rows.append({
        "transaction_id": f"TXN-{300000 + i:06d}",
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "account_id": account_id,
        "account_age_days": account_age_days,
        "ip_address": ip,
        "device_id": device,
        "sku": p2_sku,
        "seller_id": p2_seller,
        "amount": amount,
    })

# ---------- Pattern 3: Mild coordination (MEDIUM risk) ----------
# 5 accounts, shared IP, but older accounts and longer time window.
# Should score lower than patterns 1 and 2.

p3_rows = []
p3_ip = "172.16.33.99"
p3_sku = "SKU-BOOK-314"
p3_seller = "SELLER-SUS-Z"
p3_burst_start = datetime(2026, 3, 19, 10, 5, 0)

for i in range(5):
    account_id = f"ACC-GANG3-{i:02d}"
    # Older accounts: 6 to 12 days (less suspicious than pattern 1 and 2)
    account_age_days = random.randint(6, 12)
    ip = p3_ip
    device = f"DEV-SUS-{i:03d}"
    # Longer window: up to 8 minutes
    timestamp = p3_burst_start + timedelta(seconds=random.randint(0, 480))
    amount = round(random.uniform(45, 55), 2)
    p3_rows.append({
        "transaction_id": f"TXN-{400000 + i:06d}",
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "account_id": account_id,
        "account_age_days": account_age_days,
        "ip_address": ip,
        "device_id": device,
        "sku": p3_sku,
        "seller_id": p3_seller,
        "amount": amount,
    })

# ---------- combine and save ----------

df = pd.concat([
    pd.DataFrame(normal_rows),
    pd.DataFrame(p1_rows),
    pd.DataFrame(p2_rows),
    pd.DataFrame(p3_rows),
], ignore_index=True)

df = df.sort_values("timestamp").reset_index(drop=True)
df.to_csv(OUT, index=False)

total_fraud = len(p1_rows) + len(p2_rows) + len(p3_rows)
print(f"Generated {len(df)} transactions.")
print(f"  Normal background:            {len(normal_rows)}")
print(f"  Pattern 1 (IP gang):          {len(p1_rows)}")
print(f"  Pattern 2 (device rotation):  {len(p2_rows)}")
print(f"  Pattern 3 (mild coord.):      {len(p3_rows)}")
print(f"  Total fraud planted:          {total_fraud} ({100*total_fraud/len(df):.1f}%)")
print(f"Saved to: {OUT}")
