import argparse
from train_run import train_model, run_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the Mortal Kombat RL model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--run", action="store_true", help="Run the trained model")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.run:
        run_model()
    else:
        print("Please specify --train or --run")
