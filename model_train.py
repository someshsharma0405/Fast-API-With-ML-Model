
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load a built-in dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")

# 5. Save the model and its class names
model_bundle = {
    "model": model,
    "target_names": iris.target_names.tolist()
}

joblib.dump(model_bundle, "model.pkl")

print("Model successfully saved as model.pkl")
