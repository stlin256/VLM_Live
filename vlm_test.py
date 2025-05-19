import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda"

model_name = "./SmolVLM-256M-Instruct" #load from local file
#model_name = "./SmolVLM-256M-Instruct-finetuned"  # finetuned model
print(f"Loading model from {model_name}...")
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).to(DEVICE)

# Create messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Classify the situation of solar panels into one of these categories: ['Snow-Covered', 'Bird-Drop', 'Clean', 'Dusty', 'Physical-Damage', 'Electrical-Damage']. Output only the category name"}
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
    type = generated_texts[0].split()[-1]
    type = type[:len(type)-1]
    return type

def get_subfolders_and_file_dict(root_folder="./Faulty_solar_panel"):
    """
    Traverse all subfolders in the specified directory,
    create a list of subfolders names and a dictionary of file paths.
    """
    subfolders = []
    file_dict = {}
    for entry in os.listdir(root_folder):
        full_path = os.path.join(root_folder, entry)
        if os.path.isdir(full_path):
            subfolders.append(entry)
            file_dict[entry] = []
            for file_name in os.listdir(full_path):
                file_path = os.path.join(full_path, file_name)
                if os.path.isfile(file_path):
                    file_dict[entry].append(file_path)
    return subfolders, file_dict

def main():
    subfolders, file_dict = get_subfolders_and_file_dict("./Faulty_solar_panel")
    currencies = []
    all_corrects = 0
    all = 0
    for items in subfolders:
        print(f"Predicting: {items}")
        corrects = 0
        for entry in file_dict[items]:
            all += 1
            get_type = chat(load_image(entry))
            print(f"True: {items}, Predicted: {get_type}",end="")
            if get_type.lower()[:4] == items.lower()[:4]:
                corrects += 1
                print("   correct")
            else:
                print("   incorrect")
        all_corrects += corrects
        currencies.append(f"type:{items},correct__rate:{corrects/len(file_dict[items])}")
    print("--- Category Accuracy Results ---")
    for final in currencies:
        print(final)
    print("--- Overall Accuracy ---")
    print(f"Total Correct: {all_corrects}/{all}, Accuracy: {all_corrects/all:.4f}")
if __name__ == "__main__":
    main()