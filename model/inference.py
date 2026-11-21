import json
import os
import numpy as np


def model_fn(model_dir):
    """Carga el modelo entrenado y los metadatos desde JSON puro (compatible con cualquier NumPy)."""
    bundle_path = os.path.join(model_dir, "model_bundle.json")
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    bundle["user_factors"] = np.array(bundle["user_factors"], dtype=float)
    bundle["item_components"] = np.array(bundle["item_components"], dtype=float)
    return bundle


def input_fn(request_body, request_content_type="application/json"):
    """Procesa la petición JSON entrante."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        top_n = data.get("top_n", data.get("top_k", 5))
        return {
            "user_id": str(data.get("user_id", "U001")),
            "top_n": int(top_n),
        }
    raise ValueError(f"Content-Type no soportado: {request_content_type}")


def predict_fn(input_data, model_bundle):
    """Calcula recomendaciones mediante factorización matricial TruncatedSVD."""
    user_id = str(input_data.get("user_id", ""))
    top_n = int(input_data.get("top_n", input_data.get("top_k", 5)))

    user_factors = model_bundle["user_factors"]
    item_components = model_bundle["item_components"]
    user_to_idx = model_bundle["user_to_idx"]
    idx_to_item = model_bundle["idx_to_item"]
    catalog = model_bundle["catalog"]
    popular_items = model_bundle["popular_items"]

    if user_id in user_to_idx:
        u_idx = user_to_idx[user_id]
        scores = np.dot(user_factors[u_idx], item_components)
        ranked_indices = np.argsort(scores)[::-1][:top_n]
        strategy = "collaborative_svd"

        recommendations = []
        for idx in ranked_indices:
            item_id = idx_to_item.get(str(idx)) or idx_to_item.get(idx)
            product = catalog[item_id]
            recommendations.append({
                "item_id": item_id,
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "score": round(float(scores[idx]), 4),
            })
    else:
        # Estrategia Cold Start para usuarios nuevos sin historial previo
        strategy = "popularity_fallback"
        recommendations = []
        for item_id in popular_items[:top_n]:
            product = catalog[item_id]
            recommendations.append({
                "item_id": item_id,
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "score": 0.5,
            })

    return {
        "user_id": user_id,
        "strategy": strategy,
        "recommendations": recommendations,
    }


def output_fn(prediction, response_content_type="application/json"):
    """Serializa la respuesta en formato JSON."""
    if response_content_type == "application/json":
        return json.dumps(prediction, ensure_ascii=False)
    raise ValueError(f"Response Content-Type no soportado: {response_content_type}")