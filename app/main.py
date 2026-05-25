"""
app/main.py

FastAPI server with a single POST /predict endpoint.
Clients send flow features, server predicts BENIGN or DDoS.
"""

import logging
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import math
from app.predictor import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="DDoS Detection API")


class FlowFeatures(BaseModel):
    flow_duration:      float
    total_fwd_packets:  int
    flow_bytes_s:       float
    flow_pkts_s:        float
    avg_packet_size:    float
    syn_flag_count:     int
    ack_flag_count:     int

    @field_validator("flow_duration", "flow_bytes_s", "flow_pkts_s", "avg_packet_size")
    @classmethod
    def must_be_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Feature value must be finite")
        return v


class PredictionResult(BaseModel):
    label:      str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict_endpoint(flow: FlowFeatures):
    result = predict(flow.model_dump())

    if result["label"] == "DDoS":
        log.warning(
            f"⚠  DDoS detected   confidence={result['confidence']}  "
            f"pkts={flow.total_fwd_packets}  pkts_s={flow.flow_pkts_s:.1f}  "
            f"bytes_s={flow.flow_bytes_s:.1f}  avg_pkt={flow.avg_packet_size:.1f}"
        )
    else:
        log.info(
            f"✓  BENIGN           confidence={result['confidence']}  "
            f"pkts={flow.total_fwd_packets}  pkts_s={flow.flow_pkts_s:.1f}  "
            f"bytes_s={flow.flow_bytes_s:.1f}  avg_pkt={flow.avg_packet_size:.1f}"
        )

    return result