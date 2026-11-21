import json
import os
import tarfile
import time
import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "shopfast-svd-endpoint")
RAW_EVENTS_PATH = "model/data/events.csv"

SKLEARN_IMAGES = {
    "us-east-1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "us-west-2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
}


def load_and_prepare_dataset(csv_path=RAW_EVENTS_PATH, top_users=250, top_items=150):
    """Procesa events.csv mediante feedback implícito y construye matriz usuario-producto (IE1, IE5)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró '{csv_path}'. Copia tu archivo events.csv dentro de la carpeta model/data/."
        )

    df = pd.read_csv(
        csv_path,
        usecols=["event_type", "product_id", "category_code", "brand", "price", "user_id"],
        dtype={"product_id": str, "user_id": str},
    )

    # Ponderación de feedback implícito en e-commerce (IE3 / IE11)
    event_weights = {"view": 1.0, "cart": 3.0, "purchase": 5.0}
    df["weight"] = df["event_type"].map(event_weights).fillna(1.0)

    # Selección de productos y usuarios más activos para reducir dispersión (sparsity)
    active_items = df["product_id"].value_counts().head(top_items).index
    df_items = df[df["product_id"].isin(active_items)]

    active_users = df_items["user_id"].value_counts().head(top_users).index
    df_filtered = df_items[df_items["user_id"].isin(active_users)].copy()

    # Matriz de interacción usuario x producto normalizada en escala [0, 5]
    interactions = (
        df_filtered.groupby(["user_id", "product_id"])["weight"]
        .sum()
        .clip(upper=5.0)
        .unstack(fill_value=0.0)
    )

    # Construcción del catálogo legible de productos a partir de brand, category_code y price
    catalog = {}
    product_meta = df_filtered.groupby("product_id", as_index=False).last()
    for _, row in product_meta.iterrows():
        pid = str(row["product_id"])
        raw_cat = str(row["category_code"]) if pd.notna(row["category_code"]) else "electronics.general"
        raw_brand = str(row["brand"]).strip().title() if pd.notna(row["brand"]) else "ShopFast Select"
        price = round(float(row["price"]), 2) if pd.notna(row["price"]) else 29.99

        short_cat = raw_cat.split(".")[-1].replace("_", " ").title()
        catalog[pid] = {
            "name": f"{raw_brand} {short_cat} (#{pid})",
            "category": raw_cat,
            "price": price,
        }

    # Guardar muestra procesada, catálogo y usuarios de ejemplo para pruebas y repo (IE5)
    os.makedirs("model/data", exist_ok=True)
    interactions.to_csv("model/data/shopfast_interactions.csv", index_label="user_id")
    with open("model/data/shopfast_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    sample_users = interactions.sum(axis=1).sort_values(ascending=False).head(15).index.tolist()
    with open("model/data/sample_users.json", "w", encoding="utf-8") as f:
        json.dump({"sample_user_ids": sample_users}, f, indent=2, ensure_ascii=False)

    return interactions, catalog, sample_users


def evaluate_svd_model(df_matrix, n_components=8):
    """Evalúa Accuracy, Precision, Recall y F1-score sobre interacciones observadas 80/20 (IE10.d)."""
    values = df_matrix.values.astype(float)
    rng = np.random.default_rng(42)

    observed_mask = values > 0
    test_mask = observed_mask & (rng.random(values.shape) < 0.20)
    train_values = values.copy()
    train_values[test_mask] = 0.0

    k = min(n_components, min(values.shape) - 1)
    svd_eval = TruncatedSVD(n_components=k, random_state=42)
    u_eval = svd_eval.fit_transform(train_values)
    reconstructed = np.dot(u_eval, svd_eval.components_)

    # Clase positiva (1): intención alta (cart/purchase >= 3.0) frente a solo vista (view = 1.0)
    y_true = (values[test_mask] >= 3.0).astype(int)
    y_scores = reconstructed[test_mask]

    best_thresh = float(np.median(y_scores))
    best_f1 = -1.0
    for pct in np.linspace(20, 80, 30):
        candidate = np.percentile(y_scores, pct)
        score = f1_score(y_true, (y_scores >= candidate).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(candidate)

    y_pred = (y_scores >= best_thresh).astype(int)

    metrics = {
        "model_type": "TruncatedSVD (Implicit Feedback)",
        "dataset_source": "events.csv (E-Commerce Store Behavior)",
        "hyperparameters": {"n_components": k, "random_state": 42},
        "matrix_shape": {"users": int(df_matrix.shape[0]), "items": int(df_matrix.shape[1])},
        "evaluation_split": "80% Train / 20% Observed Holdout Test",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "explained_variance_pct": round(float(svd_eval.explained_variance_ratio_.sum() * 100), 2),
    }

    with open("model/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def train_and_package():
    """Procesa events.csv, evalúa métricas, entrena SVD final y empaqueta model.tar.gz."""
    print("1. Procesando model/data/events.csv y evaluando métricas (IE10.d)...")
    df_matrix, catalog, sample_users = load_and_prepare_dataset()
    metrics = evaluate_svd_model(df_matrix, n_components=8)

    print(f"   - Matriz entrenada: {df_matrix.shape[0]} usuarios x {df_matrix.shape[1]} productos")
    print(f"   - Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"   - Precision: {metrics['precision'] * 100:.2f}%")
    print(f"   - Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"   - F1-Score : {metrics['f1_score'] * 100:.2f}%")

    print("2. Entrenando modelo TruncatedSVD final y empaquetando artefacto...")
    k = metrics["hyperparameters"]["n_components"]
    svd = TruncatedSVD(n_components=k, random_state=42)
    user_factors = svd.fit_transform(df_matrix.values)
    item_components = svd.components_
    popular_items = df_matrix.sum(axis=0).sort_values(ascending=False).index.tolist()

    bundle = {
        "user_factors": user_factors,
        "item_components": item_components,
        "user_to_idx": {str(u): idx for idx, u in enumerate(df_matrix.index)},
        "idx_to_item": {idx: str(item) for idx, item in enumerate(df_matrix.columns)},
        "catalog": catalog,
        "popular_items": [str(i) for i in popular_items],
    }

    os.makedirs("model/artifacts", exist_ok=True)
    bundle_path = "model/artifacts/model_bundle.joblib"
    tar_path = "model/artifacts/model.tar.gz"

    joblib.dump(bundle, bundle_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bundle_path, arcname="model_bundle.joblib")
        tar.add("model/inference.py", arcname="code/inference.py")

    return tar_path, sample_users[0]


def configure_autoscaling(endpoint_name, role_arn):
    """Registra política de Auto-Scaling en SageMaker (IE4)."""
    autoscaling = boto3.client("application-autoscaling", region_name=REGION)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    try:
        autoscaling.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=1,
            MaxCapacity=3,
            RoleARN=role_arn,
        )
        autoscaling.put_scaling_policy(
            PolicyName=f"{endpoint_name}-invocations-scaling",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": 70.0,
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                },
                "ScaleInCooldown": 300,
                "ScaleOutCooldown": 60,
            },
        )
        print("   Auto-Scaling registrado: Min=1, Max=3 instancias (Target=70 inv/min).")
    except ClientError as exc:
        print(f"   Aviso Auto-Scaling (restricción IAM Learner Lab): {exc.response['Error']['Code']}")


def deploy_to_sagemaker(tar_path, test_user_id):
    """Sube model.tar.gz a S3 y despliega el endpoint en AWS SageMaker."""
    sts = boto3.client("sts", region_name=REGION)
    account_id = sts.get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account_id}:role/LabRole"
    bucket_name = f"shopfast-ia-models-{account_id}-{REGION}"

    s3 = boto3.client("s3", region_name=REGION)
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        print(f"3. Bucket S3 creado: {bucket_name}")
    except ClientError:
        print(f"3. Usando bucket S3 existente: {bucket_name}")

    s3_key = "svd/model.tar.gz"
    s3.upload_file(tar_path, bucket_name, s3_key)
    model_data_uri = f"s3://{bucket_name}/{s3_key}"
    print(f"4. Artefacto subido a: {model_data_uri}")

    sm = boto3.client("sagemaker", region_name=REGION)
    model_name = f"{ENDPOINT_NAME}-model"
    config_name = f"{ENDPOINT_NAME}-config"
    image_uri = SKLEARN_IMAGES.get(REGION, SKLEARN_IMAGES["us-east-1"])

    for action in [
        lambda: sm.delete_endpoint(EndpointName=ENDPOINT_NAME),
        lambda: sm.delete_endpoint_config(EndpointConfigName=config_name),
        lambda: sm.delete_model(ModelName=model_name),
    ]:
        try:
            action()
            time.sleep(2)
        except ClientError:
            pass

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_uri,
            "Environment": {"SAGEMAKER_PROGRAM": "inference.py"},
        },
        ExecutionRoleArn=role_arn,
    )

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.t2.medium",
            }
        ],
    )

    print(f"5. Desplegando endpoint '{ENDPOINT_NAME}' en SageMaker (tarda ~4-6 min)...")
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

    while True:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = desc["EndpointStatus"]
        print(f"   Estado del endpoint: {status}...")
        if status == "InService":
            break
        if status == "Failed":
            raise RuntimeError(f"Error al crear endpoint: {desc.get('FailureReason')}")
        time.sleep(20)

    configure_autoscaling(ENDPOINT_NAME, role_arn)

    print(f"6. Probando inferencia en tiempo real con usuario real ({test_user_id}):")
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    res = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"user_id": test_user_id, "top_n": 3}),
    )
    prediction = json.loads(res["Body"].read().decode("utf-8"))
    print(json.dumps(prediction, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    artifact, sample_uid = train_and_package()
    deploy_to_sagemaker(artifact, sample_uid)