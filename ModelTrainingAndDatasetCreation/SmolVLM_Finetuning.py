"""
Fine tuning for SmolVLM-256M-Instruct based on official Hugging Face resources

Resources used:
- Hugging Face GitHub (smollm): https://github.com/huggingface/smollm/blob/main/vision/finetuning/Smol_VLM_FT.ipynb
- Hugging Face Learn: https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl
"""

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import TrainingArguments, Trainer
import os

USE_LORA = True
USE_QLORA = False
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(
    model_id
)

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
            r=16, 
            lora_alpha=8, 
            lora_dropout=0.05, 
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'], 
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian" 
        )
lora_config.inference_mode = False 
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16 
    )

model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config if USE_QLORA else None,
    _attn_implementation="flash_attention_2",
    device_map="auto" 
)
model.add_adapter(lora_config)
model.enable_adapters()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]


def load_csv(csv_path, images_folder_path):
    df = pd.read_csv(csv_path, sep='\t', encoding='ISO-8859-1')
    df['image_path'] = df['IMAGE_FILE'].apply(lambda x: os.path.join(images_folder_path, x))
    return df[['image_path', 'DESCRIPTION']]


def load_dataset():
    images_folder_path = './SemArt/Images'
    train_csv_path = './SemArt/semart_train.csv'
    val_csv_path = './SemArt/semart_val.csv'
    test_csv_path = './SemArt/semart_test.csv'

    train_df = load_csv(train_csv_path, images_folder_path)
    val_df = load_csv(val_csv_path, images_folder_path)
    test_df = load_csv(test_csv_path, images_folder_path)

    return train_df, val_df, test_df


class SEMARTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert("RGB")
        description = row['DESCRIPTION']

        return {
            "image":image,
            "description":description
        }

def load_data(train_df, val_df, test_df):
    train_dataset = SEMARTDataset(train_df)
    val_dataset = SEMARTDataset(val_df)
    test_dataset = SEMARTDataset(test_df)

    return train_dataset, val_dataset, test_dataset

def collate_fn(examples):
  texts = []
  images = []
  for example in examples:
      image = example["image"]
      if image.mode != 'RGB':
        image = image.convert('RGB')
      description = example["description"]
      messages = [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": "Describe the painting in detail, including its main elements, themes and possible meaning."},
                  {"type": "image"}
              ]
          },
          {
              "role": "assistant",
              "content": [
                  {"type": "text", "text": description}
              ]
          }
      ]
      text = processor.apply_chat_template(messages, add_generation_prompt=False) 
      texts.append(text.strip())
      images.append([image])

  batch = processor(text=texts, images=images, return_tensors="pt", padding=True) 
  labels = batch["input_ids"].clone()
  labels[labels == processor.tokenizer.pad_token_id] = -100
  labels[labels == image_token_id] = -100
  batch["labels"] = labels

  return batch

def train(train_dataset):
    output_directory = "./training_logs"

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4, 
        warmup_steps=50, 
        learning_rate=1e-5,
        weight_decay=0.01, 
        logging_steps=25, 
        save_strategy="steps",
        save_steps=250, 
        save_total_limit=1,
        optim="paged_adamw_8bit", 
        bf16=True, 
        output_dir=output_directory,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    train_df, val_df, test_df = load_dataset()
    train_dataset, val_dataset, test_dataset = load_data(train_df, val_df, test_df)
    train(train_dataset)