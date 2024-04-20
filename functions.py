import datetime
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


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

# Prepare the features and target for the model
def prepare_data(user_data):
    X = np.array([[info['calories'], info['protein'], info['carbs'], info['fat'], info['activity_level']] for info in user_data])
    y = np.array([info['weight'] for info in user_data])
    return X, y


# Train ensemble model
def train_ensemble_model(X_train, y_train):
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Initialize individual models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    svr = SVR(gamma='scale', C=1.0, epsilon=0.2)
    knn = KNeighborsRegressor(n_neighbors=5)

    # Define the stacking ensemble
    estimator_list = [
        ('rf', rf),
        ('svr', svr),
        ('knn', knn)
    ]

    # Build stack model
    stack_model = StackingRegressor(estimators=estimator_list, final_estimator=LinearRegression())

    # Train stacked model
    stack_model.fit(X_train, y_train)

    return stack_model, scaler

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

# Predict weight using the trained model
def predict_weight():
    user_data = load_data()

    # Check if there are enough data points to create a model
    if len(user_data) < 2:
        print("\nNot enough data to make predictions.")
        return

    # Prepare the dataset for the model
    X, y = prepare_data(user_data)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Create a scaler object
    scaler = StandardScaler()

    # Prompt user for new dietary data to predict their weight
    print("\nEnter your today's diet information to predict future weight:")
    calories = float(input("Calories consumed: "))
    protein = float(input("Protein (g): "))
    carbs = float(input("Carbohydrates (g): "))
    fat = float(input("Fat (g): "))
    activity_level = float(input("Activity level (1-5): "))

    # Scale the input data
    X_new = scaler.fit_transform(np.array([[calories, protein, carbs, fat, activity_level]]))

    # Predict the future weight
    prediction = model.predict(X_new)
    print(f"\nPredicted weight: {prediction[0]:.2f} kg")

    # Visualize the regression line and user's progress
    plt.figure(figsize=(10, 6))

    # Plot the actual weight measurements over time
    sorted_data = sorted(user_data, key=lambda x: datetime.datetime.strptime(x['date'], "%m/%d/%Y"))
    dates = [datetime.datetime.strptime(info['date'], "%m/%d/%Y") for info in sorted_data]
    weights = [info['weight'] for info in sorted_data]
    plt.plot(dates, weights, 'bo-', label='Actual weight')

    # Plot the regression line
    X_dates = np.array([(date - dates[0]).days for date in dates]).reshape(-1, 1)  # This line needs correction
    X_regression = np.array([[calories, protein, carbs, fat, activity_level] for _ in range(len(dates))])
    regression_line = model.predict(X_regression)
    plt.plot(dates, regression_line, 'r--', label='Regression line')

    # Plot the predicted weight
    plt.plot(dates[-1] + datetime.timedelta(days=1), prediction, 'ro', label='Predicted weight')

    plt.xlabel('Date')
    plt.ylabel('Weight (kg)')
    plt.title('Weight Prediction and Progress')
    plt.legend()
    plt.grid(True)
    plt.show()


def recommend_diet_plan():
    user_data = load_data()
