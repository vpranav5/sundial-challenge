"""Quick training script with reduced parameters for CPU/Mac."""

import argparse
from pathlib import Path
from data_gen import VacationDataGenerator
from models import VacationPlannerLLM


def main():
    print("="*60)
    print("VACATION PLANNER - Quick LoRA Training (CPU-optimized)")
    print("="*60)
    
    # CPU-friendly settings
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Open model, no auth needed
    lora_r = 8  # Reduced rank
    batch_size = 1
    max_steps = 50  # Quick training
    output_dir = "./models/vacation-planner-lora"
    
    print(f"\nConfig:")
    print(f"  Model: {model_name} (1B - CPU friendly)")
    print(f"  LoRA rank: {lora_r}")
    print(f"  Steps: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Estimated time: 20-30 minutes\n")
    
    # Generate training data
    train_file = "vacation_train.json"
    val_file = "vacation_val.json"
    
    if not (Path(train_file).exists() and Path(val_file).exists()):
        print("Generating synthetic training data...")
        generator = VacationDataGenerator()
        generator.save_dataset(train_file, num_examples=100)
        generator.save_dataset(val_file, num_examples=20)
        print("✅ Data generated\n")
    else:
        print(f"Using existing data: {train_file}, {val_file}\n")
    
    # Initialize and train
    print("Loading model and adding LoRA adapters...")
    llm = VacationPlannerLLM(model_name=model_name, lora_r=lora_r)
    
    try:
        llm.load_base_model()
        print("✅ Model loaded\n")
        
        print("Starting training...")
        llm.train(train_file, val_file, output_dir, batch_size=batch_size, max_steps=max_steps)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: {output_dir}")
        print("\nTo use in demo mode:")
        print("  python main.py --demo --use-finetuned-llm")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
