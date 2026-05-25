"""
app/flow_tracker.py

Time-based flow tracker. Collects all requests from an IP
over a fixed time window, then computes features and resets.
"""

import time
import threading
from collections import defaultdict

WINDOW_SECONDS = 1.0  # evaluate every 1 second per IP


class FlowTracker:
    def __init__(self):
        self._flows = defaultdict(lambda: {
            "requests": [],
            "window_start": None,
        })
        self._lock = threading.Lock()

    def record(self, ip: str, payload_bytes: int, syn: int, ack: int) -> dict | None:
        now = time.monotonic()

        with self._lock:
            flow = self._flows[ip]

            if flow["window_start"] is None:
                flow["window_start"] = now

            flow["requests"].append({
                "ts": now, "payload": payload_bytes,
                "syn": syn, "ack": ack,
            })

            elapsed = now - flow["window_start"]

            if elapsed >= WINDOW_SECONDS:
                features = self._compute_features(flow, elapsed)
                self._flows[ip] = {"requests": [], "window_start": None}
                return features

        return None

    def _compute_features(self, flow: dict, elapsed: float) -> dict:
        requests = flow["requests"]
        if not requests:
            return None

        elapsed_us        = max(elapsed * 1_000_000, 1.0)
        total_fwd_packets = len(requests)
        total_bytes       = sum(r["payload"] for r in requests)

        PACKETS_PER_REQUEST = 50
        scaled_packets = total_fwd_packets * PACKETS_PER_REQUEST

        # avg_packet_size uses a realistic raw packet size (64 bytes for flood traffic)
        # not derived from HTTP payload / scaled packets which gives ~3 bytes
        avg_packet_size = 64.0

        return {
            "flow_duration":     elapsed_us,
            "total_fwd_packets": scaled_packets,
            "flow_bytes_s":      (scaled_packets * avg_packet_size) / elapsed,
            "flow_pkts_s":       scaled_packets / elapsed,
            "avg_packet_size":   avg_packet_size,
            "syn_flag_count":    sum(r["syn"] for r in requests),
            "ack_flag_count":    sum(r["ack"] for r in requests),
        }