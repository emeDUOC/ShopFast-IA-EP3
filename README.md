# ShopFast IA — Sistema de Recomendación Empresarial en AWS

Proyecto de implementación y despliegue en producción del motor de recomendaciones personalizadas para **ShopFast** (Asignatura ITY1102 — Arquitectura de Sistemas de IA).

## Arquitectura Desplegada
- **Modelo de Machine Learning (`AWS SageMaker`):** Filtrado colaborativo basado en factorización matricial (`TruncatedSVD`) desplegado como endpoint en tiempo real.
- **IA Generativa e Interpretabilidad (`AWS Bedrock`):** Integración con **Claude 3 Haiku** para generar explicaciones en lenguaje natural orientadas a la toma de decisiones comerciales.
- **API REST (`FastAPI`):** Exposición de endpoints documentados con Swagger/OpenAPI (`/docs`), autenticación por `X-API-Key` y *Rate Limiting*.
- **Mensajería Distribuida (`RabbitMQ`):** Procesamiento asíncrono de eventos de clics e interacciones de usuarios.
- **Dashboard Gerencial (`Streamlit`):** Interfaz visual operativa con recomendaciones en vivo, explicabilidad e impacto comercial.

## Despliegue Rápido con Docker Compose
1. Copiar `.env.example` a `.env` y completar credenciales de AWS.
2. Levantar los servicios:
   ```bash
   docker compose up --build -d
   ```
3. Accesos:
   - **Swagger UI (API):** `http://<IP-EC2>:8000/docs`
   - **Dashboard Streamlit:** `http://<IP-EC2>:8501`
   - **RabbitMQ Management:** `http://<IP-EC2>:15672`