import datetime
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# Helper functions for JSON data handling
import json
import os

def load_data(file_path='user_data.json'):
    # Check if the file exists and is not empty
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []  # Return an empty list if file doesn't exist or is empty
    else:
        with open(file_path, 'r') as file:
            return json.load(file)


def save_data(data, file_path='user_data.json'):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def validate_date(date_str):
    try:
        datetime.datetime.strptime(date_str, "%m/%d/%Y")
        return True
    except ValueError:
        return False


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
    user_data = load_data()
    if not user_data:
        print("\nNo data to provide feedback.")
        return

    total_calories = sum(info['calories'] for info in user_data)
    total_protein = sum(info['protein'] for info in user_data)
    total_carbs = sum(info['carbs'] for info in user_data)
    total_fat = sum(info['fat'] for info in user_data)

    protein_percent = (total_protein * 4 * 100) / total_calories
    carbs_percent = (total_carbs * 4 * 100) / total_calories
    fat_percent = (total_fat * 9 * 100) / total_calories

    print("\nDiet Feedback:")
    print(f"Average calories per day: {total_calories / len(user_data):.2f}")
    print(f"Protein intake: {protein_percent:.2f}%")
    print(f"Carbohydrate intake: {carbs_percent:.2f}%")
    print(f"Fat intake: {fat_percent:.2f}%")

def predict_weight():
    user_data = load_data()
    if len(user_data) < 2:
        print("\nNot enough data to make predictions.")
        return

    X = np.array([[info['calories'], info['protein'], info['carbs'], info['fat']] for info in user_data])
    y = np.array([info['weight'] for info in user_data])
    model = LinearRegression()
    model.fit(X, y)

    calories = float(input("\nEnter calories: "))
    protein = float(input("Enter protein (g): "))
    carbs = float(input("Enter carbohydrates (g): "))
    fat = float(input("Enter fat (g): "))

    prediction = model.predict(np.array([[calories, protein, carbs, fat]]))
    print(f"\nPredicted weight: {prediction[0]:.2f} kg")

def recommend_diet_plan():
    user_data = load_data()
