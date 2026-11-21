import json
import os
import sys

import pika
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("model"))

from api.main import app, generate_bedrock_pitch  # noqa: E402
from inference import input_fn, model_fn, predict_fn  # noqa: E402

load_dotenv()
client = TestClient(app)


# ==============================================================================
# BLOQUE 1: PRUEBAS UNITARIAS (Lógica aislada del modelo SVD y reglas de negocio)
# ==============================================================================


def test_unit_01_load_model_bundle():
    """CP-01 (Unitaria): Verifica la carga del artefacto JSON y matrices NumPy."""
    bundle = model_fn("model/artifacts")
    assert "user_factors" in bundle
    assert "item_components" in bundle
    assert bundle["user_factors"].ndim == 2
    assert bundle["item_components"].ndim == 2


def test_unit_02_input_fn_valid_json():
    """CP-02 (Unitaria): Valida el parseo correcto de payloads JSON entrantes."""
    payload = json.dumps({"user_id": "1515915625592213549", "top_n": 4})
    parsed = input_fn(payload, "application/json")
    assert parsed["user_id"] == "1515915625592213549"
    assert parsed["top_n"] == 4


def test_unit_03_input_fn_invalid_content_type():
    """CP-03 (Unitaria): Verifica rechazo ante Content-Type no soportado."""
    with pytest.raises(Exception):
        input_fn("user_id=123", "text/plain")


def test_unit_04_svd_known_user_ranking():
    """CP-04 (Unitaria): Comprueba recomendaciones SVD ordenadas por score para cliente conocido."""
    bundle = model_fn("model/artifacts")
    result = predict_fn({"user_id": "1515915625592213549", "top_n": 5}, bundle)
    recs = result["recommendations"]
    scores = [item["score"] for item in recs]

    assert result["strategy"] == "collaborative_svd"
    assert len(recs) == 5
    assert scores == sorted(scores, reverse=True)


def test_unit_05_cold_start_unknown_user():
    """CP-05 (Unitaria): Valida estrategia de respaldo (Cold Start) ante usuario nuevo."""
    bundle = model_fn("model/artifacts")
    result = predict_fn({"user_id": "usuario_nuevo_sin_historial_999", "top_n": 4}, bundle)

    assert result["strategy"] != "collaborative_svd"
    assert len(result["recommendations"]) == 4
    for item in result["recommendations"]:
        assert all(k in item for k in ("item_id", "name", "category", "price", "score"))


def test_unit_06_genai_pitch_generation():
    """CP-06 (Unitaria): Verifica generación de mensaje comercial personalizado."""
    sample_recs = [{"name": "Msi Videocards (#4183859)", "price": 205.25}]
    pitch = generate_bedrock_pitch("1515915625592213549", sample_recs)

    assert isinstance(pitch["message"], str)
    assert len(pitch["message"]) > 15
    assert "genai_source" in pitch


# ==============================================================================
# BLOQUE 2: PRUEBAS DE INTEGRACIÓN (FastAPI + AWS SageMaker + CloudAMQP)
# ==============================================================================


def test_integration_07_health_endpoint():
    """CP-07 (Integración): Verifica disponibilidad y configuración de GET /health."""
    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert data["sagemaker_endpoint"] == "shopfast-svd-endpoint"


def test_integration_08_metrics_endpoint():
    """CP-08 (Integración): Valida exposición de métricas ML en GET /metrics."""
    response = client.get("/metrics")
    data = response.json()

    assert response.status_code == 200
    assert isinstance(data, dict) and len(data) > 0


def test_integration_09_recommend_sagemaker_e2e():
    """CP-09 (Integración): Flujo completo POST /recommend -> SageMaker -> RabbitMQ."""
    response = client.post(
        "/recommend",
        json={"user_id": "1515915625592213549", "top_n": 5, "include_ai_pitch": True},
    )
    data = response.json()

    assert response.status_code == 200
    assert len(data["recommendations"]) == 5
    assert data["inference_source"].startswith("aws_sagemaker:")
    assert data["personalized_pitch"] is not None
    assert data["messaging_status"] == "published_to_shopfast_events"


def test_integration_10_recommend_validation_error():
    """CP-10 (Integración): Verifica validación Pydantic (422) con parámetros inválidos."""
    response = client.post("/recommend", json={"user_id": "1515915625592213549", "top_n": 0})
    assert response.status_code == 422


def test_integration_11_events_publish_rabbitmq():
    """CP-11 (Integración): Publicación de evento de compra vía POST /events a CloudAMQP."""
    response = client.post(
        "/events",
        json={"user_id": "1515915625592213549", "product_id": "4183859", "event_type": "purchase"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["queued"] is True
    assert data["status"] == "published_to_shopfast_events"


def test_integration_12_rabbitmq_broker_roundtrip():
    """CP-12 (Integración): Ciclo completo de publicación, consumo y ACK en CloudAMQP."""
    rabbitmq_url = os.getenv("RABBITMQ_URL")
    assert rabbitmq_url and rabbitmq_url.startswith("amqps://")

    test_queue = "shopfast_qa_roundtrip_test"
    params = pika.URLParameters(rabbitmq_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=test_queue, durable=False)

    test_payload = {"event": "qa_check", "status": "verified"}
    channel.basic_publish(exchange="", routing_key=test_queue, body=json.dumps(test_payload))

    method_frame, _, body = channel.basic_get(queue=test_queue, auto_ack=False)
    assert method_frame is not None
    assert json.loads(body.decode("utf-8")) == test_payload

    channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    channel.queue_delete(queue=test_queue)
    connection.close()