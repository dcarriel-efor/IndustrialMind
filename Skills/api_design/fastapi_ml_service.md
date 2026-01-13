# FastAPI ML Service Pattern

## Purpose
Standard pattern for creating FastAPI services that serve ML models with proper request validation, error handling, and monitoring.

## When to Use
- Deploying ML models as REST APIs (Month 2+)
- Real-time inference services
- Model serving endpoints

## Prerequisites
- FastAPI and Pydantic installed
- Trained model available
- Understanding of REST API design

---

## Template

```python
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class SensorReading(BaseModel):
    """Single sensor reading for inference."""
    timestamp: datetime
    machine_id: str = Field(..., min_length=1, max_length=50)
    temperature: float = Field(..., ge=-50, le=200, description="Temperature in Celsius")
    vibration: float = Field(..., ge=0, le=50, description="Vibration in mm/s")
    pressure: float = Field(..., ge=0, le=100, description="Pressure in bar")
    power: float = Field(..., ge=0, le=10000, description="Power in kW")

    @validator('timestamp')
    def timestamp_not_future(cls, v):
        if v > datetime.utcnow():
            raise ValueError('Timestamp cannot be in the future')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "machine_id": "MACHINE_001",
                "temperature": 65.5,
                "vibration": 1.2,
                "pressure": 45.0,
                "power": 250.0
            }
        }


class BatchSensorReadings(BaseModel):
    """Batch of sensor readings."""
    readings: List[SensorReading] = Field(..., min_items=1, max_items=1000)


class AnomalyPrediction(BaseModel):
    """Anomaly detection result."""
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0, description="Reconstruction error")
    threshold: float
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime
    machine_id: str


class BatchAnomalyPredictions(BaseModel):
    """Batch prediction results."""
    predictions: List[AnomalyPrediction]
    batch_size: int
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float


# ============================================================================
# FastAPI Application
# ============================================================================

class MLModelService:
    """ML Model serving with FastAPI."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_version: str = "1.0.0",
        threshold: float = 0.05,
        device: str = 'cpu'
    ):
        self.app = FastAPI(
            title="IndustrialMind Anomaly Detection API",
            description="Real-time anomaly detection for industrial sensors",
            version=model_version,
            docs_url="/docs",
            redoc_url="/redoc"
        )

        self.model = model.to(device)
        self.model.eval()
        self.model_version = model_version
        self.threshold = threshold
        self.device = device
        self.start_time = datetime.utcnow()

        # Setup routes
        self._setup_routes()

        logger.info(f"ML Service initialized (version {model_version})")

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/", tags=["General"])
        async def root():
            """Root endpoint."""
            return {
                "service": "IndustrialMind Anomaly Detection",
                "version": self.model_version,
                "docs": "/docs"
            }

        @self.app.get("/health", response_model=HealthResponse, tags=["General"])
        async def health_check():
            """Health check endpoint."""
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None,
                model_version=self.model_version,
                uptime_seconds=uptime
            )

        @self.app.post(
            "/predict",
            response_model=AnomalyPrediction,
            status_code=status.HTTP_200_OK,
            tags=["Inference"]
        )
        async def predict_single(reading: SensorReading):
            """
            Predict anomaly for a single sensor reading.

            Returns anomaly detection result with confidence score.
            """
            try:
                start_time = datetime.utcnow()

                # Prepare input
                features = torch.tensor([[
                    reading.temperature,
                    reading.vibration,
                    reading.pressure,
                    reading.power
                ]], dtype=torch.float32).to(self.device)

                # Inference
                with torch.no_grad():
                    anomaly_score = self._compute_anomaly_score(features).item()

                # Determine if anomaly
                is_anomaly = anomaly_score > self.threshold

                # Compute confidence (simple heuristic)
                confidence = min(abs(anomaly_score - self.threshold) / self.threshold, 1.0)

                inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                logger.info(
                    f"Prediction: machine={reading.machine_id}, "
                    f"anomaly={is_anomaly}, score={anomaly_score:.4f}, "
                    f"time={inference_time:.2f}ms"
                )

                return AnomalyPrediction(
                    is_anomaly=is_anomaly,
                    anomaly_score=anomaly_score,
                    threshold=self.threshold,
                    confidence=confidence,
                    timestamp=reading.timestamp,
                    machine_id=reading.machine_id
                )

            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {str(e)}"
                )

        @self.app.post(
            "/predict/batch",
            response_model=BatchAnomalyPredictions,
            status_code=status.HTTP_200_OK,
            tags=["Inference"]
        )
        async def predict_batch(batch: BatchSensorReadings):
            """
            Predict anomalies for a batch of sensor readings.

            Efficient batch inference for multiple readings.
            """
            try:
                start_time = datetime.utcnow()

                # Prepare batch input
                features_list = []
                for reading in batch.readings:
                    features_list.append([
                        reading.temperature,
                        reading.vibration,
                        reading.pressure,
                        reading.power
                    ])

                features = torch.tensor(
                    features_list,
                    dtype=torch.float32
                ).to(self.device)

                # Batch inference
                with torch.no_grad():
                    anomaly_scores = self._compute_anomaly_score(features).cpu().numpy()

                # Create predictions
                predictions = []
                for reading, score in zip(batch.readings, anomaly_scores):
                    is_anomaly = float(score) > self.threshold
                    confidence = min(abs(float(score) - self.threshold) / self.threshold, 1.0)

                    predictions.append(AnomalyPrediction(
                        is_anomaly=is_anomaly,
                        anomaly_score=float(score),
                        threshold=self.threshold,
                        confidence=confidence,
                        timestamp=reading.timestamp,
                        machine_id=reading.machine_id
                    ))

                inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                logger.info(
                    f"Batch prediction: size={len(batch.readings)}, "
                    f"time={inference_time:.2f}ms"
                )

                return BatchAnomalyPredictions(
                    predictions=predictions,
                    batch_size=len(predictions),
                    inference_time_ms=inference_time
                )

            except Exception as e:
                logger.error(f"Batch prediction error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Batch prediction failed: {str(e)}"
                )

        @self.app.exception_handler(ValueError)
        async def value_error_handler(request, exc):
            """Handle validation errors."""
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Validation error", "detail": str(exc)}
            )

    def _compute_anomaly_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score (reconstruction error for autoencoder).

        Override this method for different model types.
        """
        reconstructed = self.model(features)
        mse = torch.mean((features - reconstructed) ** 2, dim=1)
        return mse


# ============================================================================
# Application Factory
# ============================================================================

def create_app(model_path: str, threshold: float = 0.05) -> FastAPI:
    """
    Create FastAPI application with loaded model.

    Example:
        app = create_app("models/autoencoder.pt", threshold=0.05)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    from ml_models.anomaly_detector.model import SensorAutoencoder

    # Load model
    model = SensorAutoencoder(input_dim=4, latent_dim=2)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create service
    service = MLModelService(
        model=model,
        model_version="1.0.0",
        threshold=threshold,
        device='cpu'
    )

    return service.app
```

---

## Example Usage

### Running the Service

```python
# main.py
import uvicorn
from Skills.api_design.fastapi_ml_service import create_app

if __name__ == "__main__":
    # Create app
    app = create_app(
        model_path="checkpoints/autoencoder/best_checkpoint.pt",
        threshold=0.05
    )

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

### Client Usage

```python
import requests
from datetime import datetime

# Single prediction
reading = {
    "timestamp": datetime.utcnow().isoformat(),
    "machine_id": "MACHINE_001",
    "temperature": 85.0,  # High temp - likely anomaly
    "vibration": 2.5,
    "pressure": 45.0,
    "power": 250.0
}

response = requests.post(
    "http://localhost:8000/predict",
    json=reading
)

result = response.json()
print(f"Anomaly: {result['is_anomaly']}")
print(f"Score: {result['anomaly_score']:.4f}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## Common Pitfalls

### ❌ Not Setting Model to Eval Mode
```python
# Wrong
model = load_model()
output = model(input)  # Dropout still active!

# Correct
model = load_model()
model.eval()
with torch.no_grad():
    output = model(input)
```

### ❌ Not Validating Input Data
```python
# Wrong - no validation
@app.post("/predict")
def predict(data: dict):  # Any dict accepted!
    return model.predict(data)

# Correct - use Pydantic models
@app.post("/predict")
def predict(reading: SensorReading):  # Validated!
    return model.predict(reading)
```

---

*Created: 2026-01-12*
*For: IndustrialMind Month 2 - Model Serving*
*Version: 1.0*
