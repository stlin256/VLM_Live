import os
import torch
from datasets import load_dataset, IterableDataset
from PIL import Image, UnidentifiedImageError
from transformers import (
    Idefics3ForConditionalGeneration,
    AutoProcessor,
)
from trl import SFTConfig, SFTTrainer
import sys
import json
import math


# --- config ---
class ScriptConfig:
    # 1. path
    local_model_path = "./SmolVLM-256M-Instruct"
    train_json_path = "solar_panel_train_dataset.json"
    test_json_path = "solar_panel_test_dataset.json"
    output_dir = "./SmolVLM-256M-Instruct-finetuned"

    # 2. SFTConfig
    equivalent_epochs_to_train = 1  # set equivalent epochs to train
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 4
    warmup_steps = 50
    learning_rate = 3e-4
    weight_decay = 0.01
    logging_steps = 25
    save_strategy = "steps"
    save_steps = 25
    save_total_limit = 1
    optim = "adamw_torch"
    report_to = "tensorboard"
    remove_unused_columns = False
    gradient_checkpointing = True

    dataset_text_field = ""
    dataset_kwargs = {"skip_prepare_dataset": True}
    max_seq_length = 1024

    dataloader_num_workers = 1
    dataloader_pin_memory = True

    trust_remote_code_arg = True
    shuffle_buffer_size = 1000

    attempt_evaluation = True if os.path.exists(test_json_path) else False
    load_best_model_at_end = False

    max_steps: int = 0

cfg = ScriptConfig()

system_message_content = """You are a Vision Language Model specialized in interpreting visual data from solar panel images.
Your task is to analyze the provided solar panel image and respond to queries about its condition with concise answers.
The solar panels may be clean, dusty, snow-covered, have bird droppings, or exhibit physical damage.
Focus on delivering accurate, succinct answers based on the visual information. Output only the category name for classification tasks if specified."""


def convert_json_sample_to_chat_list(example_from_json):
    try:
        if not isinstance(example_from_json, dict): return None
        required_keys = ['image_path', 'question', 'answer']
        if not all(key in example_from_json for key in required_keys): return None

        pil_image = Image.open(example_from_json['image_path']).convert("RGB")
        chat_interaction = [
            {"role": "system", "content": [{"type": "text", "text": system_message_content}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": example_from_json['question']},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": example_from_json['answer']}]},
        ]
        return chat_interaction
    except (FileNotFoundError, UnidentifiedImageError):
        # print(f"Warning: Skipping sample due to FileNotFoundError or UnidentifiedImageError for {example_from_json.get('image_path', 'unknown image')}")
        return None
    except Exception as e:
        # print(f"Warning: Skipping sample due to generic error: {e} for {example_from_json.get('image_path', 'unknown image')}")
        return None


def filter_none_samples(sample):
    return sample is not None

def chat_dataset_generator(json_file_path):
    raw_dataset_stream = load_dataset("json", data_files=json_file_path, split="train", streaming=True)
    for raw_sample in raw_dataset_stream:
        chat_list = convert_json_sample_to_chat_list(raw_sample)
        if chat_list is not None:
            yield chat_list


# --- calculate max steps for equivalent epochs---
def get_max_steps_for_equivalent_epochs(
        json_file_path: str,
        equivalent_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int
) -> int:
    """
    Calculate the max_steps required for a given number of equivalent epochs.
    It will read the training JSON file, calculate the number of valid samples, and then calculate the total number of steps based on batch size and gradient accumulation.
    """
    print(f"\n--- Calculating max_steps for {equivalent_epochs} equivalent epoch(s) ---")
    if not os.path.exists(json_file_path):
        print(f"ERROR: Training JSON file '{json_file_path}' not found. Cannot calculate max_steps.")
        return 0

    num_valid_samples = 0
    print(f"Counting valid samples in '{json_file_path}'...")
    temp_dataset_stream_for_counting = load_dataset("json", data_files=json_file_path, split="train", streaming=True)

    processed_count = 0
    for raw_sample in temp_dataset_stream_for_counting:
        processed_count += 1
        if convert_json_sample_to_chat_list(raw_sample) is not None:
            num_valid_samples += 1
        if processed_count % 1000 == 0:  # Optional: progress update for large files
            print(f"  ...scanned {processed_count} raw entries, found {num_valid_samples} valid so far...")

    del temp_dataset_stream_for_counting

    print(
        f"Found {num_valid_samples} valid samples after processing {processed_count} raw entries from '{json_file_path}'.")

    if num_valid_samples == 0:
        print("Warning: No valid samples found in the training data. max_steps will be 0.")
        return 0

    num_devices = 1
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()

    print(f"Number of devices considered for step calculation: {num_devices}")

    total_samples_per_optimizer_step = per_device_train_batch_size * num_devices * gradient_accumulation_steps

    if total_samples_per_optimizer_step == 0:
        print("Warning: total_samples_per_optimizer_step is 0. Cannot calculate max_steps.")
        return 0

    steps_per_epoch = math.ceil(num_valid_samples / total_samples_per_optimizer_step)
    calculated_max_steps = steps_per_epoch * equivalent_epochs

    print(f"  - Equivalent Epochs: {equivalent_epochs}")
    print(f"  - Valid Samples: {num_valid_samples}")
    print(f"  - Per Device Batch Size: {per_device_train_batch_size}")
    print(f"  - Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"  - Number of Devices: {num_devices}")
    print(f"  - Total samples processed per optimizer step: {total_samples_per_optimizer_step}")
    print(f"  - Steps per equivalent epoch: {steps_per_epoch}")
    print(f"==> Calculated max_steps: {calculated_max_steps}")
    print(f"--- End of max_steps calculation ---\n")

    return calculated_max_steps


def main():
    cfg.max_steps = get_max_steps_for_equivalent_epochs(
        json_file_path=cfg.train_json_path,
        equivalent_epochs=cfg.equivalent_epochs_to_train,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )

    if cfg.max_steps == 0 and cfg.equivalent_epochs_to_train > 0:
        print("ERROR: Calculated max_steps is 0, but equivalent_epochs_to_train > 0. Aborting training.")
        print("Please check your training data, batch size, or GPU availability.")
        sys.exit("Exiting due to max_steps calculation resulting in 0.")
    elif cfg.max_steps == 0 and cfg.equivalent_epochs_to_train == 0:
        print("INFO: equivalent_epochs_to_train is 0, so max_steps is 0. No training will occur.")

    if not torch.cuda.is_available(): sys.exit("ERROR：GPU NOT FOUND。")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    use_bf16_training = torch.cuda.is_bf16_supported() and hasattr(torch, "bfloat16") and torch.cuda.is_bf16_supported()
    torch_dtype_model_load = torch.bfloat16 if use_bf16_training else torch.float16
    print(f"Use bfloat16 for training: {use_bf16_training}, dtype: {torch_dtype_model_load}")

    print(f"loading model: {cfg.local_model_path}")
    model_kwargs = {"trust_remote_code": cfg.trust_remote_code_arg, "torch_dtype": torch_dtype_model_load}
    model = Idefics3ForConditionalGeneration.from_pretrained(cfg.local_model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(cfg.local_model_path, trust_remote_code=cfg.trust_remote_code_arg)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        if model.config.pad_token_id != processor.tokenizer.pad_token_id:
            model.config.pad_token_id = processor.tokenizer.pad_token_id

    image_token_str = "<image>";
    image_token_id = -1
    if hasattr(processor, 'image_token_id') and processor.image_token_id is not None:
        image_token_id = processor.image_token_id
    elif hasattr(processor.tokenizer, 'image_token_id') and processor.tokenizer.image_token_id is not None:
        image_token_id = processor.tokenizer.image_token_id
    elif image_token_str in processor.tokenizer.additional_special_tokens:
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index(image_token_str)]
    else:
        vocab_tid = processor.tokenizer.convert_tokens_to_ids(image_token_str)
        if vocab_tid != processor.tokenizer.unk_token_id: image_token_id = vocab_tid
    if image_token_id != -1:
        print(f"Image Token ID: {image_token_id}")
    else:
        print(f"Warning: Image Token ID Not Found for '{image_token_str}'.")

    print("Preparing dataset...")
    train_dataset_formatted = IterableDataset.from_generator(
        chat_dataset_generator, gen_kwargs={"json_file_path": cfg.train_json_path}
    )
    train_dataset_formatted = train_dataset_formatted.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=42)
    print("Done!")

    eval_dataset_formatted = None
    if cfg.attempt_evaluation:
        try:
            temp_eval_stream = IterableDataset.from_generator(
                chat_dataset_generator, gen_kwargs={"json_file_path": cfg.test_json_path}
            )
            first_eval_item = next(iter(temp_eval_stream), None)
            del temp_eval_stream

            if first_eval_item is not None:
                eval_dataset_formatted = IterableDataset.from_generator(
                    chat_dataset_generator, gen_kwargs={"json_file_path": cfg.test_json_path}
                )
                print("Evaluating dataset preparing done!")
            else:
                print("Warning: Evaluating dataset is empty, skip evaluation.")
                eval_dataset_formatted = None
        except Exception as e:
            print(f"Fail to load evaluating dataset: {type(e).__name__} - {e}。")
            eval_dataset_formatted = None
    else:
        print("Skip evaluation.")

    def collate_fn(examples_list_of_chat_lists):
        valid_chat_lists = [cl for cl in examples_list_of_chat_lists if cl is not None]
        if not valid_chat_lists: return {}

        texts = [processor.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=False)
                 for chat_list in valid_chat_lists]

        pil_images_for_batch = []
        valid_texts_for_batch = []

        for idx, chat_list in enumerate(valid_chat_lists):
            user_content = chat_list[1]["content"];
            img_item = next((i for i in user_content if i["type"] == "image"), None)
            if img_item and "image" in img_item and isinstance(img_item["image"], Image.Image):
                pil_image = img_item["image"]
                if pil_image.mode != "RGB": pil_image = pil_image.convert("RGB")
                pil_images_for_batch.append([pil_image])
                valid_texts_for_batch.append(texts[idx])

        if not pil_images_for_batch or not valid_texts_for_batch or len(pil_images_for_batch) != len(
                valid_texts_for_batch):
            return {}

        try:
            batch = processor(text=valid_texts_for_batch, images=pil_images_for_batch, return_tensors="pt",
                              padding=True)
        except Exception as e_proc:
            print(
                f"ERROR: Processor fail in collate_fn: {e_proc}, texts len: {len(valid_texts_for_batch)}, images lists len: {len(pil_images_for_batch)}")
            return {}

        labels = batch["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        if image_token_id != -1:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    print("Setting SFTConfig...")
    sft_config_params = {
        "output_dir": cfg.output_dir,
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "warmup_steps": cfg.warmup_steps,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "logging_steps": cfg.logging_steps,
        "save_strategy": cfg.save_strategy,
        "save_total_limit": cfg.save_total_limit,
        "optim": cfg.optim,
        "bf16": use_bf16_training,
        "report_to": cfg.report_to,
        "remove_unused_columns": cfg.remove_unused_columns,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "dataset_text_field": cfg.dataset_text_field,
        "dataset_kwargs": cfg.dataset_kwargs,
        "max_seq_length": cfg.max_seq_length,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "dataloader_pin_memory": cfg.dataloader_pin_memory,
    }
    if not use_bf16_training and torch.cuda.is_available():
        sft_config_params["fp16"] = True
    if cfg.save_strategy == "steps":
        sft_config_params["save_steps"] = getattr(cfg, 'save_steps', cfg.logging_steps)

    training_sft_config = SFTConfig(**sft_config_params)

    print(
        f"SFTConfig: max_steps={training_sft_config.max_steps}, save_strategy={training_sft_config.save_strategy}, dataset_text_field='{training_sft_config.dataset_text_field}', max_seq_length={training_sft_config.max_seq_length}, dataloader_num_workers={training_sft_config.dataloader_num_workers}")
    if hasattr(training_sft_config, 'evaluation_strategy'):
        print(f"SFTConfig (TrainingArguments) 推断的 evaluation_strategy: {training_sft_config.evaluation_strategy}")

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_sft_config,
        train_dataset=train_dataset_formatted,
        eval_dataset=eval_dataset_formatted,
        data_collator=collate_fn,
    )

    if cfg.max_steps > 0:
        print("Training...")
        trainer.train()

        print("Saving model...")
        trainer.save_model(cfg.output_dir)
        processor.save_pretrained(cfg.output_dir)
        log_history = trainer.state.log_history
        if log_history:
            print("Train log history:")
            try:
                log_file_path = os.path.join(cfg.output_dir, "train_log_history.json")
                with open(log_file_path, "w") as f:
                    json.dump(log_history, f, indent=4)
                print(f"Train log history has saved at {log_file_path}")
            except Exception as e:
                print(f"Fail to save train log history: {e}")
        trainer.save_state()
        print("All Done！")
    else:
        print("max_steps is 0. Skip training.")


if __name__ == "__main__":
    main()