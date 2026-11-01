import joblib

try:
  model = joblib.load("path/to/your/model.pkl")
  print("Model loaded successfully!")
except Exception as e:
  print(f"Error loading model: {e}")