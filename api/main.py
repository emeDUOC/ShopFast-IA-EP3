import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import boto3
import pika
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath("model"))
from inference import model_fn, predict_fn  # noqa: E402

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "shopfast-svd-endpoint")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = "shopfast_events"

app = FastAPI(
    title="ShopFast IA - Motor de Recomendaciones",
    description="API de recomendaciones con AWS SageMaker (SVD), AWS Bedrock (GenAI) y RabbitMQ.",
    version="1.0.0",
)


class RecommendRequest(BaseModel):
    user_id: str = Field(default="1515915625592213549", description="ID del cliente en ShopFast")
    top_n: int = Field(default=5, ge=1, le=20, description="Cantidad de productos a recomendar")
    include_ai_pitch: bool = Field(default=True, description="Generar mensaje comercial con AWS Bedrock")


class UserEventRequest(BaseModel):
    user_id: str = Field(default="1515915625592213549")
    product_id: str = Field(default="4183859")
    event_type: str = Field(default="cart", pattern="^(view|cart|purchase)$")


def invoke_svd_recommendation(user_id: str, top_n: int) -> dict:
    """Consulta el endpoint en vivo de SageMaker con respaldo automático al artefacto local."""
    try:
        runtime = boto3.client("sagemaker-runtime", region_name=REGION)
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"user_id": user_id, "top_n": top_n}),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        result["inference_source"] = f"aws_sagemaker:{ENDPOINT_NAME}"
        return result
    except (ClientError, BotoCoreError) as exc:
        bundle_dir = "model/artifacts"
        if not os.path.exists(os.path.join(bundle_dir, "model_bundle.json")):
            raise HTTPException(status_code=503, detail=f"SageMaker no disponible y sin respaldo local: {exc}")
        local_bundle = model_fn(bundle_dir)
        result = predict_fn({"user_id": user_id, "top_n": top_n}, local_bundle)
        result["inference_source"] = "local_s3_artifact_fallback"
        return result


def generate_bedrock_pitch(user_id: str, recommendations: list[dict]) -> dict:
    """Genera un mensaje comercial personalizado usando AWS Bedrock."""
    if not recommendations:
        return {"message": "Explora nuestras novedades destacadas en ShopFast.", "genai_source": "template"}

    items_summary = ", ".join([f"{r['name']} (${r['price']})" for r in recommendations[:3]])
    prompt = (
        f"Eres el asistente comercial de ShopFast. Redacta un mensaje breve, persuasivo y en español "
        f"(máximo 2 oraciones) recomendando al usuario {user_id} estos productos seleccionados por IA: "
        f"{items_summary}. Destaca el valor técnico sin inventar descuentos."
    )

    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    candidate_models = [
        "amazon.nova-micro-v1:0",
        "amazon.titan-text-express-v1",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ]

    for model_id in candidate_models:
        try:
            response = bedrock.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 140, "temperature": 0.6},
            )
            text = response["output"]["message"]["content"][0]["text"].strip()
            return {"message": text, "genai_source": f"aws_bedrock:{model_id}"}
        except (ClientError, BotoCoreError):
            continue

    top_item = recommendations[0]
    fallback_msg = (
        f"Basado en tu actividad reciente en ShopFast, te sugerimos {top_item['name']} "
        f"(${top_item['price']}) junto con otras opciones seleccionadas para potenciar tu equipo."
    )
    return {"message": fallback_msg, "genai_source": "shopfast_nlg_fallback"}


def publish_to_rabbitmq(payload: dict) -> str:
    """Publica eventos de interacción o auditoría en RabbitMQ."""
    try:
        params = pika.URLParameters(RABBITMQ_URL)
        params.socket_timeout = 2
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(payload, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        connection.close()
        return f"published_to_{QUEUE_NAME}"
    except Exception as exc:
        return f"broker_offline_skipped ({type(exc).__name__})"


@app.get("/health", tags=["Sistema"])
def health_check():
    """Verifica el estado del servicio y la configuración activa."""
    return {
        "status": "ok",
        "sagemaker_endpoint": ENDPOINT_NAME,
        "region": REGION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics", tags=["Modelo ML"])
def get_model_metrics():
    """Retorna las métricas de evaluación (Accuracy, Precision, Recall, F1) del modelo SVD."""
    metrics_path = "model/model_metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Archivo model_metrics.json no encontrado.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/recommend", tags=["Recomendaciones"])
def recommend_products(req: RecommendRequest):
    """Obtiene recomendaciones desde SageMaker, las enriquece con Bedrock y registra evento en RabbitMQ."""
    svd_result = invoke_svd_recommendation(req.user_id, req.top_n)

    genai_output: Optional[dict] = None
    if req.include_ai_pitch:
        genai_output = generate_bedrock_pitch(req.user_id, svd_result["recommendations"])

    broker_status = publish_to_rabbitmq({
        "event": "recommendation_served",
        "user_id": req.user_id,
        "strategy": svd_result["strategy"],
        "items_count": len(svd_result["recommendations"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        **svd_result,
        "personalized_pitch": genai_output,
        "messaging_status": broker_status,
    }


@app.post("/events", tags=["Mensajería Asíncrona"])
def ingest_user_event(event: UserEventRequest):
    """Recibe eventos en tiempo real (view, cart, purchase) y los encola en RabbitMQ."""
    message = {
        "event": "user_interaction",
        "user_id": event.user_id,
        "product_id": event.product_id,
        "event_type": event.event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    status = publish_to_rabbitmq(message)
    return {"queued": status.startswith("published"), "queue": QUEUE_NAME, "status": status, "payload": message}