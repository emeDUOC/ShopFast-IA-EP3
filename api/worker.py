import json
import os
import pika
from dotenv import load_dotenv

load_dotenv()

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = "shopfast_events"


def callback(ch, method, properties, body):
    event = json.loads(body.decode("utf-8"))
    print(f"[RabbitMQ Worker] Evento procesado: {json.dumps(event, ensure_ascii=False)}")
    ch.basic_ack(delivery_tag=method.delivery_tag)


def start_worker():
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

    print(f"Escuchando eventos en cola '{QUEUE_NAME}' (Presiona Ctrl+C para salir)...")
    channel.start_consuming()


if __name__ == "__main__":
    start_worker()