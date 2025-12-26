import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

print("Enter student details:")

# Take input from user
study_hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
score = float(input("Previous exam score: "))

# Convert input to model format (2D array)
input_data = np.array([[study_hours, attendance, score]])

# Predict
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

# Show result
print("\nPrediction Result:")
if prediction[0] == 1:
    print(" PASS")
else:
    print(" FAIL")
confidence = max(probability[0]) * 100
