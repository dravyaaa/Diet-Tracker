# functions.py
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import json

# Data structures to store diet information
diet_data = []
user_data = []

def validate_date(date_str):
    try:
        datetime.datetime.strptime(date_str, "%m/%d/%Y")
        return True
    except ValueError:
        return False

def enter_diet_info():
    date_valid = False
    while not date_valid:
        date_str = input("\nEnter date (mm/dd/yyyy): ")
        if validate_date(date_str):
            date_valid = True
        else:
            print("Invalid date format. Please enter the date in the format mm/dd/yyyy.")

    activity_level_valid = False
    while not activity_level_valid:
        try:
            activity_level = int(input("Enter your daily activity level (1-5): "))
            if activity_level < 1 or activity_level > 5:
                print("\nInvalid input. Please enter a number between 1 and 5.")
            else:
                activity_level_valid = True
        except ValueError:
            print("\nInvalid input. Please enter a number between 1 and 5.")

    calories_valid = False
    while not calories_valid:
        try:
            calories = int(input("Enter calories consumed: "))
            calories_valid = True
        except ValueError:
            print("\nInvalid input. Please enter a valid number for calories consumed.")

    protein_valid = False
    while not protein_valid:
        try:
            protein = float(input("Enter protein (g): "))
            protein_valid = True
        except ValueError:
            print("\nInvalid input. Please enter a valid number for protein.")

    carbs_valid = False
    while not carbs_valid:
        try:
            carbs = float(input("Enter carbohydrates (g): "))
            carbs_valid = True
        except ValueError:
            print("\nInvalid input. Please enter a valid number for carbohydrates.")

    fat_valid = False
    while not fat_valid:
        try:
            fat = float(input("Enter fat (g): "))
            fat_valid = True
        except ValueError:
            print("\nInvalid input. Please enter a valid number for fat.")

    weight_valid = False
    while not weight_valid:
        try:
            weight = float(input("Enter current weight (kg): "))
            weight_valid = True
        except ValueError:
            print("\nInvalid input. Please enter a valid number for weight.")

    diet_info = {
        "date": date_str,
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
        "weight": weight,
        "activity_level": activity_level
    }

    diet_data.append(diet_info)
    user_data.append({"calories": calories, "protein": protein, "carbs": carbs, "fat": fat, "activity_level": activity_level})
    print("\nDiet information recorded successfully.")

def display_data():
    if not diet_data:
        print("\nNo data to display.")
        return

    print("\nDiet Data:")
    for info in diet_data:
        print(f"\nDate: {info['date']}")
        print(f"Calories: {info['calories']}")
        print(f"Protein: {info['protein']} g")
        print(f"Carbohydrates: {info['carbs']} g")
        print(f"Fat: {info['fat']} g")
        print(f"Weight: {info['weight']} kg")
        print(f"Activity Level: {info['activity_level']}")
        print("-" * 20)

def get_feedback():
    if not diet_data:
        print("\nNo data to provide feedback.")
        return

    # Calculate macronutrient percentages
    total_calories = sum(info['calories'] for info in diet_data)
    total_protein = sum(info['protein'] for info in diet_data)
    total_carbs = sum(info['carbs'] for info in diet_data)
    total_fat = sum(info['fat'] for info in diet_data)

    protein_percent = (total_protein * 4 * 100) / total_calories
    carbs_percent = (total_carbs * 4 * 100) / total_calories
    fat_percent = (total_fat * 9 * 100) / total_calories

    print("\nDiet Feedback:")
    print(f"\nAverage calories per day: {total_calories / len(diet_data):.2f}")
    print(f"Protein intake: {protein_percent:.2f}% (Recommended: 10-35%)")
    print(f"Carbohydrate intake: {carbs_percent:.2f}% (Recommended: 45-65%)")
    print(f"Fat intake: {fat_percent:.2f}% (Recommended: 20-35%)")

    if protein_percent < 10 or protein_percent > 35:
        print("\nConsider adjusting your protein intake to meet the recommended range.")
    if carbs_percent < 45 or carbs_percent > 65:
        print("\nConsider adjusting your carbohydrate intake to meet the recommended range.")
    if fat_percent < 20 or fat_percent > 35:
        print("\nConsider adjusting your fat intake to meet the recommended range.")

def predict_weight():
    if len(diet_data) < 2:
        print("\nNot enough data to make predictions.")
        return

    # Prepare data for linear regression
    X = np.array([[info['calories'], info['protein'], info['carbs'], info['fat'], info['activity_level']] for info in diet_data])
    y = np.array([info['weight'] for info in diet_data])

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)

    activity_level_valid = False
    while not activity_level_valid