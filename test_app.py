
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    response = client.post("/predict", json={
        "features": {
            "OverallQual": 7,
            "GrLivArea": 1710,
            "GarageCars": 2,
            "TotalBsmtSF": 856,
            "YearBuilt": 2003
        }
    })

    assert response.status_code == 200
    assert "predicted_sale_price" in response.json()
