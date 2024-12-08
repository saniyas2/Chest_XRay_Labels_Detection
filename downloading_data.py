import os
import subprocess

os.environ['PATH'] += os.pathsep + os.path.expanduser('~/.local/bin')

# Set your Kaggle credentials
os.environ['KAGGLE_USERNAME'] = "saniya28"  # Replace with your username
os.environ['KAGGLE_KEY'] = "eade87b826cc1b04b15ada17c8ad97f3"  # Replace with your API key

# Define the target output directory
output_dir = "/home/ubuntu/new_NLP/CV_Project/data/"  # Adjust the path to your 'data' folder

# Use Kaggle CLI to download the dataset
try:
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "nih-chest-xrays/data", "-p", output_dir],
        check=True
    )
    print(f"Dataset downloaded to {output_dir}")
except FileNotFoundError:
    print("Error: Kaggle CLI not installed. Please install it with 'pip install kaggle'.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while downloading the dataset: {e}")
