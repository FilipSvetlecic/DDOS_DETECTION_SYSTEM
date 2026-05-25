"""
scripts/normal_client.py

Simulates normal traffic by sampling feature values from the real
BENIGN distribution of the CICIDS2017 dataset (p25-p75 range with noise).
"""

import time
import random
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SERVER = "http://server:8000"


def sample_benign() -> dict:
    """
    Sample a feature vector from the real BENIGN distribution.
    Ranges taken directly from CICIDS2017 Friday DDoS CSV percentiles.
    """
    return {
        # Flow Duration: wide range, mostly short-to-medium flows (µs)
        "flow_duration": random.uniform(48.0, 7_522_023.0),

        # Total Fwd Packets: typically very low for benign (1-5)
        "total_fwd_packets": int(random.choice([1, 1, 1, 2, 2, 3, 4, 5])),

        # Flow Bytes/s: moderate throughput
        "flow_bytes_s": random.uniform(0.0, 115_384.0),

        # Flow Packets/s: low to moderate rate
        "flow_pkts_s": random.uniform(0.18, 10_025.0),

        # Average Packet Size: varied, often small
        "avg_packet_size": random.uniform(0.0, 272.0),

        # SYN Flag Count: rarely set in benign flows
        "syn_flag_count": random.choices([0, 1], weights=[95, 5])[0],

        # ACK Flag Count: sometimes set
        "ack_flag_count": random.choices([0, 1], weights=[60, 40])[0],
    }


def run():
    log.info("Normal client started — sending BENIGN traffic to %s", SERVER)
    session = requests.Session()

    while True:
        features = sample_benign()

        try:
            resp = session.post(
                f"{SERVER}/predict",
                json=features,
                timeout=5,
            )
            data = resp.json()
            log.info("Sent BENIGN sample → server says: %s (%.4f)",
                    data["label"], data["confidence"])

        except requests.exceptions.ConnectionError:
            log.warning("Server unreachable, retrying in 3s...")
            time.sleep(3)
            continue

        time.sleep(random.uniform(1.0, 3.0))


if __name__ == "__main__":
    run()