from .base_model import BaseModel
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel(BaseModel):

    def __init__(self):
        super().__init__("Decision Tree")
        self.model = DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        )

    # def train(self):
    #     self.model.fit(X_train, y_train)

    # def predict(self):
    #     y_pred = self.model.predict(X_test)
    #     y_prob = self.model.predict_proba(X_test)[:, 1]
    #     return y_pred, y_prob

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        return y_pred, y_prob
