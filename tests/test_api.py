from fastapi.testclient import TestClient

from customer_churn.api.main import app, get_encoder, get_model


class FakeModel:
    def predict(self, features):
        return [1]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


class FakeEncoder:
    def transform(self, df):
        return [[0, 0, 0] for _ in range(len(df))]

    def get_feature_names_out(self, cols):
        return ["fake_a", "fake_b", "fake_c"]


app.dependency_overrides[get_model] = lambda: FakeModel()
app.dependency_overrides[get_encoder] = lambda: FakeEncoder()

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
}


def test_predict_valid_request_returns_200():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_shape():
    response = client.post("/predict", json=VALID_CUSTOMER)
    body = response.json()
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_uses_injected_fake_model():
    response = client.post("/predict", json=VALID_CUSTOMER)
    body = response.json()
    assert body["churn_prediction"] == "Yes"
    assert body["churn_probability"] == 0.8


def test_predict_missing_field_returns_422():
    bad_customer = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    bad_customer = {**VALID_CUSTOMER, "tenure": "not_a_number"}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422
