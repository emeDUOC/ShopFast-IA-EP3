# ShopFast IA — Sistema de Recomendación Empresarial en AWS

Proyecto de implementación y despliegue en producción del motor de recomendaciones personalizadas para **ShopFast** (Asignatura ITY1102 — Arquitectura de Sistemas de IA).

---

## 1. Arquitectura Desplegada

```text
+-------------------+       HTTP / REST       +------------------------------------+
|  Cliente / Swagger| ----------------------> |           FastAPI Backend          |
|  (/docs)          | <---------------------- |  (/recommend, /events, /health)    |
+-------------------+      JSON + NLG Pitch   +---------+---------------+----------+
                                                        |               |
                              AWS SDK (boto3)           |               | AMQPS (pika)
                              InvokeEndpoint            |               | publish
                                                        v               v
                                              +-------------------+   +-------------------+
                                              |   AWS SageMaker   |   | RabbitMQ Broker   |
                                              | shopfast-svd-     |   | (CloudAMQP)       |
                                              | endpoint (SVD)    |   | Cola: shopfast_   |
                                              +-------------------+   | events            |
                                                                      +---------+---------+
                                                                                |
                                                                                | consume / ACK
                                                                                v
                                                                      +-------------------+
                                                                      |  Worker Asíncrono |
                                                                      |  (api/worker.py)  |
                                                                      +-------------------+
```

### Componentes del Sistema
* **Modelo de Machine Learning (`AWS SageMaker`):** Filtrado colaborativo basado en factorización matricial (`TruncatedSVD`) desplegado como endpoint en tiempo real (`shopfast-svd-endpoint`), con estrategia de respaldo (*Cold Start*) para usuarios sin historial.
* **IA Generativa e Interpretabilidad (`AWS Bedrock`):** Integración con **Claude 3 Haiku** para generar explicaciones comerciales en lenguaje natural (con respaldo automático `shopfast_nlg_fallback`).
* **API REST (`FastAPI`):** Exposición de endpoints documentados con Swagger/OpenAPI (`/docs`) y validación de esquemas mediante Pydantic.
* **Mensajería Distribuida (`RabbitMQ / CloudAMQP`):** Procesamiento asíncrono de eventos de recomendaciones e interacciones de usuarios (`view`, `cart`, `purchase`) a través de la cola persistente `shopfast_events` y el consumidor `api/worker.py`.
* **Aseguramiento de Calidad (`Pytest`):** Suite automatizada de **12 casos de prueba** (6 unitarias y 6 de integración).

---

## 2. Estructura del Proyecto

```text
shopfast-ia-ep3/
├── api/
│   ├── main.py                # Endpoints REST (/recommend, /events, /health, /metrics)
│   └── worker.py              # Consumidor asíncrono de eventos en RabbitMQ
├── model/
│   ├── artifacts/             # Artefactos entrenados (model_bundle.json, metrics.json)
│   ├── train_svd.py           # Entrenamiento y evaluación del modelo SVD
│   ├── inference.py           # Lógica de inferencia para SageMaker (model_fn, input_fn, predict_fn)
│   └── deploy_sagemaker.py    # Empaquetado y despliegue del endpoint en AWS SageMaker
├── tests/
│   └── test_shopfast.py       # 12 pruebas unitarias y de integración (pytest)
├── .env                       # Credenciales y variables de entorno (ignorado por Git)
├── .gitignore                 # Protección de claves y archivos temporales
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación técnica del proyecto
```

---

## 3. Instalación y Ejecución

### Configuración Local (Entorno Virtual Python)
1. **Activar el entorno virtual e instalar dependencias:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Configurar variables de entorno en `.env` (en la raíz del proyecto):**
   ```env
   AWS_ACCESS_KEY_ID=tu_access_key
   AWS_SECRET_ACCESS_KEY=tu_secret_key
   AWS_SESSION_TOKEN=tu_session_token
   AWS_DEFAULT_REGION=us-east-1

   SAGEMAKER_ENDPOINT_NAME=shopfast-svd-endpoint
   BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
   RABBITMQ_URL=amqps://usuario:password@host.rmq.cloudamqp.com/vhost
   API_KEY_SECRET=shopfast-secret-key-2026
   ```
3. **Levantar los servicios en dos terminales paralelas:**
   * **Terminal 1 (API FastAPI):**
     ```powershell
     .\.venv\Scripts\uvicorn.exe api.main:app --reload
     ```
     *Swagger UI disponible en:* `[http://127.0.0.1:8000/docs]`
   * **Terminal 2 (Consumidor RabbitMQ):**
     ```powershell
     .\.venv\Scripts\python.exe api/worker.py
     ```

### Despliegue Alternativo con Docker Compose
1. Copiar `.env.example` a `.env` y completar credenciales de AWS.
2. Levantar los servicios:
   ```bash
   docker compose up --build -d
   ```

---

## 4. Endpoints Principales

| Método | Ruta | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Verifica la disponibilidad del servicio y la conexión configurada con AWS SageMaker. |
| `GET` | `/metrics` | Expone las métricas de evaluación del modelo SVD (`Accuracy`, `Precision`, `Recall`, `F1-Score`). |
| `POST` | `/recommend` | Invoca `shopfast-svd-endpoint` en AWS SageMaker, genera el texto comercial y publica el evento en RabbitMQ. |
| `POST` | `/events` | Registra interacciones de usuario (`view`, `cart`, `purchase`) en la cola `shopfast_events`. |

---

## 5. Resultados de Pruebas Unitarias e Integración (`pytest`)

Comando ejecutado:
```powershell
.\.venv\Scripts\pytest.exe -v
```
**Resultado:** `12 passed in 11.22s`

| ID | Tipo | Caso de Prueba (`pytest`) | Escenario / Entrada | Resultado Esperado | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CP-01** | Unitaria | `test_unit_01_load_model_bundle` | Carga de artefactos desde `model/artifacts/model_bundle.json` vía `model_fn`. | Reconstrucción correcta de matrices 2D `user_factors` e `item_components`. | **PASSED** |
| **CP-02** | Unitaria | `test_unit_02_input_fn_valid_json` | Serialización de payload JSON válido con `ContentType: application/json`. | Parseo exitoso conservando `user_id` y `top_n`. | **PASSED** |
| **CP-03** | Unitaria | `test_unit_03_input_fn_invalid_content_type` | Envío de datos en formato no soportado (`text/plain`) a `input_fn`. | Lanzamiento controlado de excepción por tipo de contenido inválido. | **PASSED** |
| **CP-04** | Unitaria | `test_unit_04_svd_known_user_ranking` | Inferencia en `predict_fn` para usuario histórico (`1515915625592213549`). | Estrategia `collaborative_svd` y lista ordenada descendentemente por `score`. | **PASSED** |
| **CP-05** | Unitaria | `test_unit_05_cold_start_unknown_user` | Inferencia para usuario nuevo sin interacciones previas (`usuario_nuevo_...`). | Activación automática de respaldo (`popular_fallback`) con esquema íntegro. | **PASSED** |
| **CP-06** | Unitaria | `test_unit_06_genai_pitch_generation` | Generación de mensaje comercial personalizado para producto recomendado. | Cadena de texto no vacía con etiqueta `genai_source` válida. | **PASSED** |
| **CP-07** | Integración | `test_integration_07_health_endpoint` | Petición HTTP `GET /health` sobre la API FastAPI. | Código `200 OK` y confirmación de `sagemaker_endpoint` activo. | **PASSED** |
| **CP-08** | Integración | `test_integration_08_metrics_endpoint` | Petición HTTP `GET /metrics` para consulta de evaluación del modelo. | Código `200 OK` con diccionario de métricas (`Accuracy`, `Recall`, `F1`). | **PASSED** |
| **CP-09** | Integración | `test_integration_09_recommend_sagemaker_e2e` | Flujo completo `POST /recommend` → AWS SageMaker → CloudAMQP. | Código `200 OK`, 5 ítems recomendados, `aws_sagemaker` y cola publicada. | **PASSED** |
| **CP-10** | Integración | `test_integration_10_recommend_validation_error` | Petición `POST /recommend` con parámetro fuera de rango (`top_n=0`). | Rechazo automático mediante Pydantic con código `422 Unprocessable Entity`. | **PASSED** |
| **CP-11** | Integración | `test_integration_11_events_publish_rabbitmq` | Registro de evento de compra (`purchase`) mediante `POST /events`. | Código `200 OK` y encolamiento confirmado (`queued: true`) en RabbitMQ. | **PASSED** |
| **CP-12** | Integración | `test_integration_12_rabbitmq_broker_roundtrip` | Publicación, lectura (`basic_get`) y confirmación (`basic_ack`) en CloudAMQP. | Recepción íntegra del JSON publicado y limpieza de cola temporal. | **PASSED** |