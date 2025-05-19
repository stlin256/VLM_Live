import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import json

DEVICE = "cuda"

# model_name = "./SmolVLM-256M-Instruct" #load from local file (source model)
model_name = "./SmolVLM-256M-Instruct-finetuned"  # finetuned model
print(f"Loading model from {model_name}...")
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).to(DEVICE)

TEST_JSON_FILE = "solar_panel_test_dataset.json"
#This fixed classification problem must be completely consistent with the 'fixed_classic_question' in the JSON generation script
#And it is also the text used for messages in the test script
FIXED_CLASSIFICATION_QUESTION = "Classify the situation of solar panels into one of these categories: ['Snow-Covered', 'Bird-Drop', 'Clean', 'Dusty', 'Physical-Damage', 'Electrical-Damage']. Output only the category name"

# Create messages - Use text consistent with the classification problem in the JSON generated script
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": FIXED_CLASSIFICATION_QUESTION}
        ]
    },
]

# Prepare inputs
def chat(image):
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=32)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    # The last word output by the model is the category, which may have a punctuation mark at the end
    if generated_texts and generated_texts[0]:
        parts = generated_texts[0].strip().split()
        if not parts:  # if output is empty
            return ""

        type_pred = parts[-1]  # get last word
        if type_pred and not type_pred[-1].isalnum() and len(type_pred) > 1:
            type_pred = type_pred[:-1]
        elif type_pred and not type_pred[-1].isalnum() and len(type_pred) == 1:
            return ""
        return type_pred
    return ""

def main():
    # --- load data form json file ---
    if not os.path.exists(TEST_JSON_FILE):
        print(f"Error: Test JSON file not found at {TEST_JSON_FILE}")
        return

    with open(TEST_JSON_FILE, 'r', encoding='utf-8') as f:
        test_dataset_from_json = json.load(f)

    test_files_by_true_category = {}
    processed_image_paths_for_classification = set()

    print(f"Loading and filtering test data from {TEST_JSON_FILE}...")
    for item in test_dataset_from_json:
        # We only care about the problem samples used for classification in JSON
        if item.get("question") == FIXED_CLASSIFICATION_QUESTION:
            image_path = item["image_path"]
            true_category = item["answer"]

            if image_path not in processed_image_paths_for_classification:
                if true_category not in test_files_by_true_category:
                    test_files_by_true_category[true_category] = []
                test_files_by_true_category[true_category].append(image_path)
                processed_image_paths_for_classification.add(image_path)

    if not test_files_by_true_category:
        print(
            f"No classification samples found in {TEST_JSON_FILE} matching the question:\n'{FIXED_CLASSIFICATION_QUESTION}'")
        print("Or no unique image paths were successfully added for testing.")
        return

    true_categories_in_testset = list(test_files_by_true_category.keys())

    total_unique_images = len(processed_image_paths_for_classification)
    print(
        f"Found {total_unique_images} unique images for classification across {len(true_categories_in_testset)} true categories.")

    currencies = []
    overall_correct_predictions = 0
    overall_processed_images = 0

    for true_category_name in true_categories_in_testset:
        print(f"Identifying category: {true_category_name}")
        corrects_for_category = 0
        images_in_category = test_files_by_true_category[true_category_name]


        num_images_processed_for_category = 0
        for image_file_path in images_in_category:
            if not os.path.exists(image_file_path):
                print(f"  Warning: Image file not found at {image_file_path}. Skipping.")
                continue

            try:
                image = load_image(image_file_path)
            except Exception as e:
                print(f"  Error loading image {image_file_path}: {e}. Skipping.")
                continue

            predicted_type = chat(image)
            overall_processed_images += 1
            num_images_processed_for_category += 1

            if not predicted_type:
                print(f"  Image: {os.path.basename(image_file_path)}, True: {true_category_name}, Predicted: [EMPTY/INVALID]",
                      end="")
                print("   incorrect (empty prediction)")
                continue

            print(f"  Image: {os.path.basename(image_file_path)}, True: {true_category_name}, Predicted: {predicted_type}",
                  end="")

            # Discriminant logic: Compare the first 4 characters of the predicted result with the first 4 characters of the true category (ignoring capitalization)
            if predicted_type.strip().lower()[:4] == true_category_name.strip().lower()[:4]:
                corrects_for_category += 1
                overall_correct_predictions += 1
                print("   correct")
            else:
                print("   incorrect")

        if num_images_processed_for_category > 0:
            accuracy = corrects_for_category / num_images_processed_for_category
            currencies.append(
                f"type:{true_category_name},correct_rate:{accuracy:.4f} ({corrects_for_category}/{num_images_processed_for_category})")
        else:
            currencies.append(f"type:{true_category_name},correct_rate:N/A (0 valid images processed)")

    print("\n--- Category Accuracy Results ---")
    for final_result_str in currencies:
        print(final_result_str)

    if overall_processed_images > 0:
        overall_accuracy = overall_correct_predictions / overall_processed_images
        print(f"\n--- Overall Accuracy ---")
        print(
            f"Total Correct: {overall_correct_predictions}/{overall_processed_images}, Accuracy: {overall_accuracy:.4f}")
    else:
        print("\nNo images were processed overall.")


if __name__ == "__main__":
    main()