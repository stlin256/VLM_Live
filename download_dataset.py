#this script is used to downloading dataset form kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("pythonafroz/solar-panel-images", path="./")

print("Path to dataset files:", path)


