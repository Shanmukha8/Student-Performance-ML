# Student Performance Prediction - Beginner Friendly Version
# This project predicts student exam scores based on study hours and attendance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("="*60)
print("Student Performance Prediction System")
print("="*60)

# ========================================
# STEP 1: CREATE DATASET
# ========================================
print("\n[STEP 1] Creating student dataset...")

# We'll create fake data for 150 students
np.random.seed(42)  # This ensures we get same results every time

# Generate random study hours (between 1-10 hours per day)
study_hours = np.random.uniform(1, 10, 150)

# Generate random attendance (between 40-100%)
attendance = np.random.uniform(40, 100, 150)

# Calculate performance score
# Logic: More study hours + higher attendance = better performance
# We add some randomness to make it realistic
performance = (study_hours * 5) + (attendance * 0.3) + np.random.normal(0, 5, 150)

# Make sure scores are between 0-100
performance = np.clip(performance, 0, 100)

# Put everything in a table (DataFrame)
data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Performance': performance
})

print(f"✓ Created data for {len(data)} students")
print("\nFirst 5 students:")
print(data.head())

print("\nDataset Statistics:")
print(data.describe().round(2))

# ========================================
# STEP 2: PREPARE DATA FOR TRAINING
# ========================================
print("\n[STEP 2] Preparing data for machine learning...")

# Split features (X) and target (y)
X = data[['Study_Hours', 'Attendance']]  # Input features
y = data['Performance']  # What we want to predict

# Split into training (80%) and testing (20%) sets
# Training data = used to teach the model
# Testing data = used to check if model learned correctly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training data: {len(X_train)} students")
print(f"✓ Testing data: {len(X_test)} students")

# ========================================
# STEP 3: TRAIN THE MODEL
# ========================================
print("\n[STEP 3] Training the Linear Regression model...")

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)  # This is where "learning" happens

print("✓ Model trained successfully!")

# Show what the model learned
print(f"\nModel Formula:")
print(f"Performance = {model.intercept_:.2f} + "
      f"({model.coef_[0]:.2f} × Study_Hours) + "
      f"({model.coef_[1]:.2f} × Attendance)")

# ========================================
# STEP 4: TEST THE MODEL
# ========================================
print("\n[STEP 4] Testing model accuracy...")

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✓ R² Score (Accuracy): {r2:.4f} ({r2*100:.1f}%)")
print(f"✓ RMSE (Average Error): {rmse:.2f} points")

print("\nWhat this means:")
if r2 >= 0.85:
    print("→ Excellent! Model predicts very accurately")
elif r2 >= 0.70:
    print("→ Good! Model makes decent predictions")
else:
    print("→ Fair. Model could be improved")

# ========================================
# STEP 5: VISUALIZE RESULTS
# ========================================
print("\n[STEP 5] Creating visualizations...")

# Create 4 charts to show results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Student Performance Prediction Results', 
             fontsize=16, fontweight='bold')

# Chart 1: Study Hours vs Performance
axes[0, 0].scatter(X_test['Study_Hours'], y_test, 
                   color='blue', alpha=0.6, label='Actual Scores')
axes[0, 0].scatter(X_test['Study_Hours'], y_pred, 
                   color='red', alpha=0.6, label='Predicted Scores')
axes[0, 0].set_xlabel('Study Hours per Day', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Performance Score', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Study Hours vs Performance')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Chart 2: Attendance vs Performance
axes[0, 1].scatter(X_test['Attendance'], y_test, 
                   color='blue', alpha=0.6, label='Actual Scores')
axes[0, 1].scatter(X_test['Attendance'], y_pred, 
                   color='red', alpha=0.6, label='Predicted Scores')
axes[0, 1].set_xlabel('Attendance %', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Performance Score', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Attendance vs Performance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Chart 3: Actual vs Predicted (shows accuracy)
axes[1, 0].scatter(y_test, y_pred, color='green', alpha=0.6)
axes[1, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction Line')
axes[1, 0].set_xlabel('Actual Performance', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Predicted Performance', fontsize=11, fontweight='bold')
axes[1, 0].set_title(f'Prediction Accuracy (R² = {r2:.3f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Chart 4: Prediction Errors
errors = y_test - y_pred
axes[1, 1].scatter(y_pred, errors, color='purple', alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Performance', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Prediction Error', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Error Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualizations created!")

# ========================================
# STEP 6: MAKE SAMPLE PREDICTIONS
# ========================================
print("\n[STEP 6] Testing with sample students...")
print("="*60)

# Test with 3 different student profiles
test_students = [
    {'Study_Hours': 2, 'Attendance': 50},   # Poor student
    {'Study_Hours': 5, 'Attendance': 75},   # Average student
    {'Study_Hours': 9, 'Attendance': 95},   # Excellent student
]

for i, student in enumerate(test_students, 1):
    prediction = model.predict([[student['Study_Hours'], 
                                 student['Attendance']]])[0]
    
    print(f"\nStudent {i}:")
    print(f"  Study Hours: {student['Study_Hours']} hours/day")
    print(f"  Attendance: {student['Attendance']}%")
    print(f"  → Predicted Score: {prediction:.1f}/100")
    
    if prediction >= 80:
        print(f"  → Grade: A (Excellent!)")
    elif prediction >= 60:
        print(f"  → Grade: B (Good)")
    else:
        print(f"  → Grade: C (Needs Improvement)")

print("\n" + "="*60)
print("Project Completed Successfully! ✓")
print("="*60)

# ========================================
# KEY TAKEAWAYS FOR INTERVIEW
# ========================================
print("\n📚 KEY POINTS TO REMEMBER FOR INTERVIEW:")
print("-" * 60)
print("1. What is this project?")
print("   → Predicts student exam scores using study hours & attendance")
print("\n2. What algorithm did you use?")
print("   → Linear Regression (simplest ML algorithm)")
print("\n3. How accurate is your model?")
print(f"   → {r2*100:.1f}% accuracy (R² score)")
print("\n4. What does the model learn?")
print("   → That more study hours = higher scores")
print("   → That better attendance = higher scores")
print("\n5. Libraries used:")
print("   → pandas (data handling)")
print("   → scikit-learn (machine learning)")
print("   → matplotlib (visualization)")
print("-" * 60)