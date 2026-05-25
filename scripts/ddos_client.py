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
BATCH_LOG = 10


def sample_ddos() -> dict:
    if random.random() < 0.5:
        avg_packet_size = random.uniform(7.2, 8.0)
    else:
        avg_packet_size = random.uniform(897.0, 1661.0)

    return {
        "flow_duration": random.uniform(613_298.0, 9_387_683.0),
        "total_fwd_packets": int(random.uniform(3.0, 8.0)),
        "flow_bytes_s": random.uniform(7.0, 304_955.0),
        "flow_pkts_s": random.uniform(0.5, 214.8),
        "avg_packet_size": avg_packet_size,
        "syn_flag_count": random.choices([0, 1], weights=[99, 1])[0],
        "ack_flag_count": random.choices([0, 1], weights=[30, 70])[0],
    }


def flood(thread_id: int):
    req_count = 0
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
            req_count +=1
            if req_count % BATCH_LOG == 0:
                log.info(
                "Thread %d | total: %d",
                thread_id, req_count,
            )
        except requests.exceptions.ConnectionError:
            log.warning("Thread %d - server unreachable, retrying...", thread_id)
            time.sleep(1)


def run():
    log.info("DDoS client started - flooding %s with %d threads", SERVER, THREADS)
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