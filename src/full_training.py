from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import json
import torch

# Model configuration
model_name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
max_seq_length = 2048
dtype = None

print("🦥 Loading model and tokenizer...")
# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

print("✅ Model loaded successfully!")

# Add LoRA adapters
print("🔧 Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank - higher = more capacity, more memory
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,  # LoRA scaling factor (usually 2x rank)
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
    random_state=3407,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None, # LoftQ
)

print("✅ LoRA adapters added successfully!")

# Load training data
def load_training_data():
    """Load and prepare training data from JSON file"""
    with open('training_data.json', 'r') as f:
        data = json.load(f)
    
    # Convert to the format expected by SFTTrainer
    formatted_data = []
    for item in data:
        # Combine input and output into a single text field
        text = f"### Input:\n{item['input']}\n\n### Output:\n{item['output']}"
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

print("📊 Loading training data...")
dataset = load_training_data()
print(f"✅ Loaded {len(dataset)} training examples")

# Training arguments optimized for Unsloth
print("⚙️ Configuring training arguments...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
    ),
)

print("✅ SFTTrainer configured successfully!")
print(f"\n🎯 Training Configuration:")
print(f"- Model: {model_name}")
print(f"- LoRA rank: 64")
print(f"- Batch size: 2 per device")
print(f"- Gradient accumulation: 4 steps (effective batch size: 8)")
print(f"- Epochs: 3")
print(f"- Learning rate: 2e-4")
print(f"- Warmup steps: 10")
print(f"- Training examples: {len(dataset)}")
print(f"- Output directory: outputs")

print(f"\n🚀 Ready to start training!")
print(f"Run: trainer.train() to begin fine-tuning")
