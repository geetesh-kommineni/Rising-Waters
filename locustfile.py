from locust import HttpUser, task, between

class FloodPredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def view_home(self):
        self.client.get("/")

    @task(2)
    def view_predict_form(self):
        self.client.get("/predict")

    @task(4)
    def run_prediction(self):
        payload = {
            "Temp": "29.0",
            "Humidity": "70.0",
            "Cloud Cover": "35.0",
            "ANNUAL": "3200.0",
            "Jan-Feb": "70.0",
            "Mar-May": "380.0",
            "Jun-Sep": "2100.0",
            "Oct-Dec": "660.0",
            "avgjune": "270.0",
            "sub": "650.0"
        }
        self.client.post("/predict", data=payload)
