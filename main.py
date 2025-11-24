import os

def main():
    print("\n==============================")
    print("  HEART DISEASE PREDICTION")
    print("==============================\n")
    print("Choose an option:")
    print("1. Train the Model")
    print("2. Run the Web App")
    print("3. Exit")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == "1":
        print("\n🔧 Training the Heart Disease Model...\n")
        os.system("python src/train_model.py")

    elif choice == "2":
        print("\n🚀 Launching Streamlit Web App...\n")
        os.system("streamlit run website/app.py")

    elif choice == "3":
        print("\nExiting... Goodbye!")
        exit()

    else:
        print("\n❌ Invalid option. Please run the program again.")

if __name__ == "__main__":
    main()
