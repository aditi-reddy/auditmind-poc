"""
generate_mock_data.py
Generates a realistic e-commerce transaction dataset with a planted 刷单 (fake order) gang.
Output: data/transactions.csv (50 rows: 40 normal + 10 coordinated fraud)

The planted fraud gang follows real 刷单 patterns:
- 10 accounts created within 3 days of each other
- All sharing one IP address
- Half sharing one device ID
- All buying the same SKU within a 5-minute window
- All pushing revenue to the same seller
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")

# ---------- normal background traffic (40 rows) ----------

normal_rows = []
base_time = datetime(2026, 3, 15, 9, 0, 0)

for i in range(40):
    account_id = f"ACC-{1000 + i:04d}"
    # Account age: 30 to 500 days old (established users)
    account_age_days = random.randint(30, 500)
    ip = f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
    device = f"DEV-{random.randint(10000, 99999)}"
    sku = f"SKU-{random.choice(['BOOK','SHOE','PHONE','LAMP','BAG','MUG','TOY','HAT','PEN','CUP'])}-{random.randint(100,999)}"
    seller = f"SELLER-{random.randint(1, 30):03d}"
    timestamp = base_time + timedelta(
        days=random.randint(0, 29),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    amount = round(random.uniform(15, 300), 2)
    normal_rows.append({
        "transaction_id": f"TXN-{10000 + i:05d}",
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "account_id": account_id,
        "account_age_days": account_age_days,
        "ip_address": ip,
        "device_id": device,
        "sku": sku,
        "seller_id": seller,
        "amount": amount,
    })

# ---------- planted fraud gang (10 rows) ----------
# Classic 刷单 signature: 10 young accounts, same IP, overlapping devices,
# same SKU, same seller, all within a 5-minute burst.

fraud_rows = []
shared_ip = "192.168.10.42"
shared_device = "DEV-FRAUD-001"
target_sku = "SKU-PREMWATCH-7829"
target_seller = "SELLER-FRAUD-X"
burst_start = datetime(2026, 3, 28, 23, 47, 0)  # late night burst

for i in range(10):
    account_id = f"ACC-GANG-{i:02d}"
    # Very young accounts: 2 to 5 days old
    account_age_days = random.randint(2, 5)
    # All use shared IP
    ip = shared_ip
    # Half use the shared device, the other half use rotated devices
    device = shared_device if i < 5 else f"DEV-FRAUD-{i:03d}"
    timestamp = burst_start + timedelta(seconds=random.randint(0, 270))  # within 4.5 min
    amount = round(random.uniform(480, 520), 2)  # inflated, similar amounts
    fraud_rows.append({
        "transaction_id": f"TXN-{20000 + i:05d}",
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "account_id": account_id,
        "account_age_days": account_age_days,
        "ip_address": ip,
        "device_id": device,
        "sku": target_sku,
        "seller_id": target_seller,
        "amount": amount,
    })

# ---------- combine and save ----------

df = pd.concat([pd.DataFrame(normal_rows), pd.DataFrame(fraud_rows)], ignore_index=True)
df = df.sort_values("timestamp").reset_index(drop=True)
df.to_csv(OUT, index=False)

print(f"Generated {len(df)} transactions.")
print(f"  Normal:  {len(normal_rows)}")
print(f"  Planted gang: {len(fraud_rows)}")
print(f"Saved to: {OUT}")
