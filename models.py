"""
ML model for vacation planning:
- LoRA fine-tuned LLM (Llama/TinyLlama) for natural language generation
"""

import os
# Disable MPS before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from typing import List, Tuple
from dataclasses import dataclass
import json


class VacationPlannerLLM:
    """LoRA fine-tuned Llama for vacation planning text generation."""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", lora_r=16, lora_alpha=32):
        self.model_name = model_name
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_base_model(self):
        """Load base model with 4-bit quantization and add LoRA adapters."""
        print(f"Loading {self.model_name}...")
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add LoRA adapters
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def train(self, train_data_path: str, val_data_path: str, output_dir: str = "./models/vacation-planner-lora",
              num_epochs: int = 3, batch_size: int = 4, max_steps: int = 100):
        """Fine-tune with LoRA."""
        if self.model is None:
            self.load_base_model()
        
        # Load datasets
        with open(train_data_path) as f:
            train_data = json.load(f)
        with open(val_data_path) as f:
            val_data = json.load(f)
        
        train_dataset = VacationDataset(train_data, self.tokenizer)
        val_dataset = VacationDataset(val_data, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            max_steps=max_steps,
            logging_steps=10,
            eval_steps=25,
            save_steps=50,
            eval_strategy="steps",  # Changed from evaluation_strategy
            fp16=torch.cuda.is_available(),
            optim="paged_adamw_8bit",
            report_to="none",
        )
        
        # Train
        trainer = Trainer(model=self.model, args=training_args, 
                         train_dataset=train_dataset, eval_dataset=val_dataset)
        print("Starting LoRA fine-tuning...")
        trainer.train()
        
        # Save
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def load_finetuned(self, model_path: str):
        """Load fine-tuned model."""
        print(f"Loading fine-tuned model from {model_path}")
        
        # Read base model name from adapter config
        import json
        from pathlib import Path
        config_path = Path(model_path) / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", self.model_name)
        else:
            base_model_name = self.model_name
        
        print(f"Loading base model: {base_model_name}")
        # Force CPU device to avoid MPS issues on Mac
        self.device = "cpu"
        
        # Disable MPS backend if on Mac
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        base_model = base_model.to("cpu")
        
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model = self.model.to("cpu")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self.model
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate text."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure model is on CPU (avoid MPS issues on Mac)
        device = "cpu"
        self.model = self.model.to(device)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, temperature=temperature,
                                         do_sample=True, top_p=0.9, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class VacationDataset(Dataset):
    """Dataset for vacation conversations."""
    
    def __init__(self, conversations: List[dict], tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in conversation["messages"]])
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                                  padding="max_length", return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }
