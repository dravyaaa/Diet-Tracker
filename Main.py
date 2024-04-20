# main.py
import functions
import re

def print_menu():
    print("Diet Tracking Menu")
    print("------------------")
    print("1. Enter diet information for the day")
    print("2. Display all data")
    print("3. Get feedback for staying on track")
    print("4. Predict future weight")
    print("5. Get a personalized weight loss plan")
    print("6. Exit")


def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ")
        if not choice.isdigit() or int(choice) < 1 or int(choice) > 6:
            print("\nInvalid choice. Please enter a number from 1 to 6.")
            continue

        choice = int(choice)
        if choice == 1:
            functions.enter_diet_info()
        elif choice == 2:
            functions.display_data()
        elif choice == 3:
            functions.get_feedback()
        elif choice == 4:
            print("\nPredicting weight using enhanced model...")
            print("Predicted weight:", functions.predict_weight())
        elif choice == 5:
            print("\nGenerating personalized weight loss plan...")
            plan = functions.recommend_weight_loss_plan()
            print("\nRecommended Weight Loss Plan:")
            for key, value in plan.items():
                print(f"{key}: {value}")
        elif choice == 6:
            print("\nExiting...")
            break
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
