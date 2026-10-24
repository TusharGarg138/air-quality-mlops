from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_validation_rejection():
    # Negative PM2.5 must be rejected by Pydantic schema
    invalid_payload = {
        "pm2_5": -10.0,
        "pm10": 50.0,
        "no2": 20.0,
        "so2": 10.0,
        "co": 1.0,
        "temperature": 25.0,
        "humidity": 50.0,
        "wind_speed": 2.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422