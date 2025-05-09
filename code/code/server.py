from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
from typing import List
import os
import random
import uuid
import tempfile
import base64

app = FastAPI(title="XGBoost Multi-Model Server")


class PredictRequest(BaseModel):
    features: List[float]


class AddModelRequest(BaseModel):
    model_json: str  # Base64-encoded model JSON string


class ModelManager:
    def __init__(self):
        self.models = []  # List of (model_id, xgb.Booster)

    async def load_model_from_json_string(self, model_json_str: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(model_json_str.encode('utf-8'))
            tmp_path = tmp.name

        model = xgb.Booster()
        model.load_model(tmp_path)
        os.remove(tmp_path)

        model_id = f"model_{uuid.uuid4().hex[:8]}"
        self.models.append((model_id, model))
        return model_id

    async def drop_model(self, model_id: str):
        self.models = [(mid, m) for (mid, m) in self.models if mid != model_id]

    async def predict(self, features: List[float]) -> float:
        if not self.models:
            raise ValueError("No models loaded.")

        model_id, model = random.choice(self.models)  # Randomly choose a model
        dmatrix = xgb.DMatrix(np.array([features]))
        prob = model.predict(dmatrix)[0]
        return prob, model_id


model_manager = ModelManager()


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        prob, model_id = await model_manager.predict(request.features)
        return {"probability": prob, "model_used": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_model")
async def add_model(request: AddModelRequest):
    try:
        model_json_str = request.model_json
        model_id = await model_manager.load_model_from_json_string(model_json_str)
        return {"status": "Model loaded successfully.", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/drop_model/{model_id}")
async def drop_model(model_id: str):
    try:
        await model_manager.drop_model(model_id)
        return {"status": f"Model '{model_id}' dropped successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))