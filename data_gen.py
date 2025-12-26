import pandas as pd
import numpy as np

np.random.seed(42)

# Generate features for Pass class (result=1)
hours_studied_pass = np.random.uniform(5, 10, 250)     # More hours studied
attendance_pass = np.random.uniform(75, 100, 250)      # Higher attendance
score_pass = hours_studied_pass * 7 + \
    attendance_pass * 0.5 + np.random.normal(0, 5, 250)
score_pass = np.clip(score_pass, 0, 100)

# Generate features for Fail class (result=0)
hours_studied_fail = np.random.uniform(0, 5, 250)      # Fewer hours studied
attendance_fail = np.random.uniform(40, 75, 250)       # Lower attendance
score_fail = hours_studied_fail * 7 + \
    attendance_fail * 0.5 + np.random.normal(0, 5, 250)
score_fail = np.clip(score_fail, 0, 100)

# Create DataFrames
df_pass = pd.DataFrame({
    'hours_studied': hours_studied_pass,
    'attendance': attendance_pass,
    'score': score_pass,
    'result': 1
})

df_fail = pd.DataFrame({
    'hours_studied': hours_studied_fail,
    'attendance': attendance_fail,
    'score': score_fail,
    'result': 0
})

# Combine and shuffle
df = pd.concat([df_pass, df_fail], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('balanced_student_data_500.csv', index=False)
print("Balanced dataset with 500 samples created successfully!")
