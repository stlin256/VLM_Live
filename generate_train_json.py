import os
import json
import random

def get_subfolders_and_file_dict(root_folder="./Faulty_solar_panel"):
    """
    Traverse all subfolders in the specified directory,
    create a list of subfolders names and a dictionary of file paths.
    """
    subfolders = []
    file_dict = {}
    if not os.path.isdir(root_folder):
        print(f"Error: Root folder '{root_folder}' not found.")
        return subfolders, file_dict
    for entry in os.listdir(root_folder):
        full_path = os.path.join(root_folder, entry)
        if os.path.isdir(full_path):
            subfolders.append(entry)
            file_dict[entry] = []
            for file_name in os.listdir(full_path):
                file_path = os.path.join(full_path, file_name)
                if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_dict[entry].append(file_path)
    return subfolders, file_dict

# --- Config ---
dataset_input_root_folder = "Faulty_solar_panel"
#The proportion of the training set in the total dataset (e.g. 0.8 represents 80% of the training set and 20% of the testing set)
train_set_ratio = 0.8

# output files
output_train_json_file = "solar_panel_train_dataset.json"
output_test_json_file = "solar_panel_test_dataset.json"

'''
 Dynamic question list (for visual description)
 The following content is specific to the solar panel dataset.
 If you want to transfer and use it, you need to carefully modify it, which has a great impact on the training effect of the model
'''
descriptive_questions_templates = [
    "Visually describe the condition of the solar panel in this image.",
    "What is the primary visual state of the solar panel shown?",
    "Based on the image, identify any visible issues or the status of the solar panel.",
    "Provide a visual assessment of the solar panel in the picture.",
    "What visual characteristics do you observe on this solar panel's surface?",
    "How would you classify the visual state of this solar panel?",
    "Briefly, what does this solar panel look like?",
    "Analyze the visual appearance of the solar panel in the image and report its status.",
    "Does this solar panel appear clean, dusty, or damaged? Describe its visual features.",
    "What does the image visually reveal about the solar panel's condition?"
]

# Pre set "detail" answers for each category, with a focus on visual description
category_details = {
    "Bird-drop": "The solar panel surface shows distinct, often white or dark, irregular splatters or patches, indicative of bird droppings.",
    "Clean": "The solar panel surface appears smooth, uniform in color and texture, and free of any visible obstructions or coatings. It reflects light clearly.",
    "Dusty": "The solar panel surface is covered with a fine, particulate layer, appearing dull, hazy, or less reflective than a clean panel. The original color might be obscured.",
    "Physical-Damage": "Visible signs of physical harm are present on the solar panel, such as cracks, shatters, chipped edges, broken glass, or bent framework. The surface integrity is compromised.",
    "Snow-Covered": "The solar panel is partially or fully obscured by a white, opaque layer of snow, which may have a crystalline or powdery texture.",
    "Electrical-Damage": "The solar panel exhibits electrical faults shown by brown/dark scorch marks, often in lines or round spots, or sections with unusual color changes."
}
default_detail_template = "The solar panel is visually in a '{category}' state. Observe the image for specific visual cues defining this condition."

# Fixed classification problem
fixed_classification_question = "Classify the situation of solar panels into one of these categories: ['Snow-Covered', 'Bird-Drop', 'Clean', 'Dusty', 'Physical-Damage', 'Electrical-Damage']. Output only the category name"


# --- main ---
def generate_and_split_training_data():
    all_qna_pairs = [] # Store Q&A pairs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_folder_for_get_function = os.path.join(".", dataset_input_root_folder)

    subfolders, file_dict = get_subfolders_and_file_dict(root_folder_for_get_function)

    if not subfolders and not file_dict:
        print(f"No data found using get_subfolders_and_file_dict with root: {root_folder_for_get_function}")
        return

    print(f"Starting dataset generation. Categories found: {subfolders}")

    for category_name, image_paths_in_category in file_dict.items():
        if not image_paths_in_category:
            print(f"No image files found in category: {category_name}")
            continue

        print(f"Processing category: {category_name} (found {len(image_paths_in_category)} images)...")
        for image_full_path in image_paths_in_category:
            relative_image_path = os.path.normpath(image_full_path)

            # 1. Add dynamic descriptive Q&A pairs
            for question_text in descriptive_questions_templates:
                detail_text = category_details.get(category_name, default_detail_template.format(category=category_name))
                descriptive_answer = f"Visually, the solar panel is classified as '{category_name}'. {detail_text}"
                all_qna_pairs.append({
                    "image_path": relative_image_path,
                    "question": question_text,
                    "answer": descriptive_answer
                })

            # 2. Add fixed category Q&A pairs
            all_qna_pairs.append({
                "image_path": relative_image_path,
                "question": fixed_classification_question,
                "answer": category_name
            })

    if not all_qna_pairs:
        print("No Q&A pairs were generated. Exiting.")
        return

    # Shuffle all Q&A pairs
    random.shuffle(all_qna_pairs)

    total_pairs = len(all_qna_pairs)
    train_size = int(total_pairs * train_set_ratio)
    test_size = total_pairs - train_size

    train_dataset = all_qna_pairs[:train_size]
    test_dataset = all_qna_pairs[train_size:]

    print(f"\nTotal Q&A pairs generated: {total_pairs}")
    print(f"Training set size: {len(train_dataset)} ({train_set_ratio*100:.1f}%)")
    print(f"Test set size: {len(test_dataset)} ({(1-train_set_ratio)*100:.1f}%)")

    # Save train dataset
    output_train_file_path = os.path.join(script_dir, output_train_json_file)
    with open(output_train_file_path, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=4, ensure_ascii=False)
    print(f"Training data saved to: {output_train_file_path}")

    # Save test dataset
    if test_dataset: # only save when the dataset is not empty
        output_test_file_path = os.path.join(script_dir, output_test_json_file)
        with open(output_test_file_path, 'w', encoding='utf-8') as f:
            json.dump(test_dataset, f, indent=4, ensure_ascii=False)
        print(f"Test data saved to: {output_test_file_path}")
    else:
        print("Test dataset is empty, not saving a test file.")


if __name__ == "__main__":
    generate_and_split_training_data()