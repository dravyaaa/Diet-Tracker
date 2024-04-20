from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import json
import os
import datetime
import joblib
from joblib import dump
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


# Load the updated data
def load_data(file_path='user_data.json'):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return []

def save_data(data, file_path='user_data.json'):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def validate_date(date_str):
    try:
        datetime.datetime.strptime(date_str, "%m/%d/%Y")
        return True
    except ValueError:
        return False


# Load user data from JSON file
user_data = load_data('/mnt/data/user_data.json')


def enter_diet_info():
    user_data = load_data()

    # Date input with validation
    date_str = input("\nEnter date (mm/dd/yyyy): ")
    while not validate_date(date_str):
        print("Invalid date format. Please enter the date in the format mm/dd/yyyy.")
        date_str = input("Enter date (mm/dd/yyyy): ")

    # Activity level input with validation
    while True:
        try:
            activity_level = int(input("Enter your activity level (1-5): "))
            if 1 <= activity_level <= 5:
                break
            else:
                print("Invalid activity level. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Caloric intake input with validation
    while True:
        try:
            calories = float(input("Enter calories consumed: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for calories.")

    # Protein input with validation
    while True:
        try:
            protein = float(input("Enter protein (g): "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for protein.")

    # Carbohydrates input with validation
    while True:
        try:
            carbs = float(input("Enter carbohydrates (g): "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for carbohydrates.")

    # Fat input with validation
    while True:
        try:
            fat = float(input("Enter fat (g): "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for fat.")

    # Weight input with validation
    while True:
        try:
            weight = float(input("Enter current weight (kg): "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for weight.")

    # Append the new entry
    user_data.append({
        "date": date_str,
        "activity_level": activity_level,
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
        "weight": weight
    })

    save_data(user_data)
    print("\nDiet information recorded successfully.")


def display_data():
    user_data = load_data()
    if not user_data:
        print("\nNo data to display.")
        return

    print("\nDiet Data:")
    for info in user_data:
        print(f"\nDate: {info['date']}")
        print(f"Calories: {info['calories']}")
        print(f"Protein: {info['protein']} g")
        print(f"Carbohydrates: {info['carbs']} g")
        print(f"Fat: {info['fat']} g")
        print(f"Weight: {info['weight']} kg")
        print("-" * 20)

def get_feedback():
    # Load the user data
    user_data = load_data()

    # Check if there is any data to provide feedback
    if not user_data:
        print("\nNo data to provide feedback.")
        return

    # Calculate macronutrient percentages
    total_calories = sum(info['calories'] for info in user_data)
    total_protein = sum(info['protein'] for info in user_data)
    total_carbs = sum(info['carbs'] for info in user_data)
    total_fat = sum(info['fat'] for info in user_data)

    if total_calories == 0:  # Prevent division by zero if total_calories is zero
        print("Insufficient data to calculate macronutrient percentages.")
        return

    protein_percent = (total_protein * 4 * 100) / total_calories
    carbs_percent = (total_carbs * 4 * 100) / total_calories
    fat_percent = (total_fat * 9 * 100) / total_calories

    print("\nDiet Feedback:")
    print(f"\nAverage calories per day: {total_calories / len(user_data):.2f}")
    print(f"Protein intake: {protein_percent:.2f}% (Recommended: 10-35%)")
    print(f"Carbohydrate intake: {carbs_percent:.2f}% (Recommended: 45-65%)")
    print(f"Fat intake: {fat_percent:.2f}% (Recommended: 20-35%)")

    if protein_percent < 10 or protein_percent > 35:
        print("\nConsider adjusting your protein intake to meet the recommended range.")
    if carbs_percent < 45 or carbs_percent > 65:
        print("\nConsider adjusting your carbohydrate intake to meet the recommended range.")
    if fat_percent < 20 or fat_percent > 35:
        print("\nConsider adjusting your fat intake to meet the recommended range.")

def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='model.pkl'):
    return joblib.load(filename)

def log_prediction(date, actual, predicted):
    with open('predictions_log.txt', 'a') as file:
        file.write(f"{date}: Actual Weight - {actual}, Predicted Weight - {predicted}\n")


# Predict weight using the trained model
def predict_weight():

    # Load and prepare data
    user_data = load_data()
    data = np.array(
        [[d['activity_level'], d['calories'], d['protein'], d['carbs'], d['fat'], d['weight']] for d in user_data])
    X, y = data[:, :-1], data[:, -1]

    # Train model
    model = RandomForestRegressor(max_depth=3, n_estimators=50, random_state=42)
    model.fit(X, y)
    dump(model, 'rf_model.pkl')  # Save the model

    # Predict the next day's weight
    next_day_data = X[-1] * np.random.uniform(0.95, 1.05, size=X.shape[1])
    prediction = model.predict([next_day_data])[0]

    # Plot results
    dates = [datetime.datetime.strptime(d['date'], "%m/%d/%Y") for d in user_data]
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y, 'bo-', label='Historical Weights')
    plt.plot(dates[-1] + datetime.timedelta(days=1), prediction, 'ro', label='Predicted Weight')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.title('Weight Prediction and Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

    return prediction

def recommend_diet_plan():
    user_data = load_data()
