import os
import joblib

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)   

class BaseModel:

    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = None
        self.model_path = os.path.join(
            MODEL_DIR,
            name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        )

    def save(self):
        joblib.dump({"model": self.model, "scaler": self.scaler}, self.model_path)

    def load(self):
        package = joblib.load(self.model_path)
        self.model = package["model"]
        self.scaler = package["scaler"]
