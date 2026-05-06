from clean_data import main as clean_data_main
from train_by_genre import train_all_genre_models


def main():
    print("Step 1: Cleaning data...")
    clean_data_main()

    print("\nStep 2: Training genre-specific models...")
    train_all_genre_models()

    print("\nProject pipeline complete.")


if __name__ == "__main__":
    main()