"""
scripts/ddos_client.py

Simulates DDoS attack traffic by sampling feature values from the real
DDoS distribution of the CICIDS2017 dataset (p25-p95 range with noise).
Sends requests as fast as possible with multiple threads.
"""

import time
import random
import requests
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SERVER  = "http://server:8000"
THREADS = 5
BATCH_LOG = 50

_count_lock = threading.Lock()
_req_count  = 0


def sample_ddos() -> dict:
    """
    Sample a feature vector from the real DDoS distribution.
    Ranges taken directly from CICIDS2017 Friday DDoS CSV percentiles.

    Key differences from BENIGN:
    - avg_packet_size: bimodal — either tiny (7-8) or very large (900-1660)
    - flow_bytes_s: much lower than you'd expect (slow, persistent flows)
    - total_fwd_packets: low (3-8), DDoS flows are short but numerous
    - SYN flags: almost never set (this is UDP/HTTP flood, not SYN flood)
    - ACK flags: more commonly set than BENIGN
    """
    # avg_packet_size is bimodal in DDoS: tiny packets OR large packets
    if random.random() < 0.5:
        avg_packet_size = random.uniform(7.2, 8.0)    # tiny flood packets
    else:
        avg_packet_size = random.uniform(897.0, 1661.0)  # large payload floods

    return {
        # Flow Duration: medium-to-long flows (µs)
        "flow_duration": random.uniform(613_298.0, 9_387_683.0),

        # Total Fwd Packets: low per flow (3-8), but thousands of flows
        "total_fwd_packets": int(random.uniform(3.0, 8.0)),

        # Flow Bytes/s: surprisingly low — DDoS here is persistent not bursty
        "flow_bytes_s": random.uniform(7.0, 304_955.0),

        # Flow Packets/s: low-to-moderate (DDoS in this file is not high-pps)
        "flow_pkts_s": random.uniform(0.5, 214.8),

        # Average Packet Size: bimodal — see above
        "avg_packet_size": avg_packet_size,

        # SYN Flag Count: almost always 0 (UDP/HTTP flood, not SYN flood)
        "syn_flag_count": random.choices([0, 1], weights=[99, 1])[0],

        # ACK Flag Count: more commonly 1 in DDoS flows
        "ack_flag_count": random.choices([0, 1], weights=[30, 70])[0],
    }


def flood(thread_id: int):
    global _req_count
    session = requests.Session()

    while True:
        features = sample_ddos()

        try:
            resp = session.post(
                f"{SERVER}/predict",
                json=features,
                timeout=2,
            )
            data = resp.json()

            with _count_lock:
                _req_count += 1
                if _req_count % BATCH_LOG == 0:
                    log.info(
                        "Thread %d | total: %d | last: %s (%.4f)",
                        thread_id, _req_count,
                        data["label"], data["confidence"]
                    )

        except requests.exceptions.ConnectionError:
            log.warning("Thread %d — server unreachable, retrying...", thread_id)
            time.sleep(1)


def run():
    log.info("DDoS client started — flooding %s with %d threads", SERVER, THREADS)
    threads = [
        threading.Thread(target=flood, args=(i,), daemon=True)
        for i in range(THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    run()