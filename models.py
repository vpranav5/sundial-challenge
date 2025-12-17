"""
ML models for vacation planning:
1. LoRA fine-tuned LLM (Llama 3.2 3B)
2. Neural preference ranker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from dataclasses import dataclass
import json


@dataclass
class VacationOption:
    """Vacation option for ranking."""
    destination: str
    hotel_category: str
    activities: List[str]
    total_cost: float
    description: str


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
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self.model
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate text."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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


class PreferenceRanker(nn.Module):
    """Neural network for ranking vacation options."""
    
    def __init__(self, embedding_dim=384, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.1)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder.eval()
    
    def forward(self, option_a: torch.Tensor, option_b: torch.Tensor) -> torch.Tensor:
        """Predict probability that user prefers option_a over option_b."""
        combined = torch.cat([option_a, option_b], dim=-1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))
    
    def encode_option(self, option: VacationOption) -> torch.Tensor:
        """Encode vacation option as embedding."""
        description = f"Destination: {option.destination}. Hotel: {option.hotel_category}. " \
                     f"Activities: {', '.join(option.activities)}. Cost: ${option.total_cost}. {option.description}"
        with torch.no_grad():
            embedding = self.encoder.encode(description, convert_to_tensor=True)
        return embedding
    
    def rank_options(self, options: List[VacationOption]) -> List[Tuple[VacationOption, float]]:
        """Rank options by preference score."""
        if len(options) <= 1:
            return [(opt, 1.0) for opt in options]
        
        embeddings = [self.encode_option(opt) for opt in options]
        scores = torch.zeros(len(options))
        
        for i in range(len(options)):
            for j in range(len(options)):
                if i != j:
                    with torch.no_grad():
                        pref_score = self.forward(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                    scores[i] += pref_score
        
        scores = scores / (len(options) - 1)
        return sorted(zip(options, scores.tolist()), key=lambda x: x[1], reverse=True)
    
    def update_from_feedback(self, preferred: VacationOption, rejected: VacationOption, 
                            optimizer: torch.optim.Optimizer) -> float:
        """Online learning from user feedback."""
        emb_preferred = self.encode_option(preferred).unsqueeze(0)
        emb_rejected = self.encode_option(rejected).unsqueeze(0)
        
        pref_score = self.forward(emb_preferred, emb_rejected)
        target = torch.ones_like(pref_score)
        loss = nn.functional.binary_cross_entropy(pref_score, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
