from .base_model import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LogisticModel(BaseModel):

    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.scaler = StandardScaler()

    # def train(self):
    #     X_train_scaled = self.scaler.fit_transform(X_train)
    #     self.model.fit(X_train_scaled, y_train)

    # def predict(self):
    #     X_test_scaled = self.scaler.transform(X_test)
    #     y_pred = self.model.predict(X_test_scaled)
    #     y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
    #     return y_pred, y_prob
    
    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        return y_pred, y_prob


