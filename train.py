"""Training script for LoRA fine-tuning."""

import argparse
from pathlib import Path
from data_gen import VacationDataGenerator
from models import VacationPlannerLLM


def main():
    parser = argparse.ArgumentParser(description="Train vacation planner with LoRA")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Base model")
    parser.add_argument("--steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output-dir", default="./models/vacation-planner-lora", help="Output directory")
    args = parser.parse_args()
    
    print("="*60)
    print("VACATION PLANNER - LoRA FINE-TUNING")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Model: {args.model}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}\n")
    
    # Generate training data
    train_file = "vacation_train.json"
    val_file = "vacation_val.json"
    
    if not (Path(train_file).exists() and Path(val_file).exists()):
        print("Generating synthetic training data...")
        generator = VacationDataGenerator()
        generator.save_dataset(train_file, num_examples=150)
        generator.save_dataset(val_file, num_examples=30)
        print("✅ Data generated\n")
    else:
        print(f"Using existing data: {train_file}, {val_file}\n")
    
    # Initialize and train
    print("Loading model and adding LoRA adapters...")
    llm = VacationPlannerLLM(model_name=args.model, lora_r=args.lora_r)
    
    try:
        llm.load_base_model()
        print("✅ Model loaded\n")
        
        print("Starting training (1-2 hours)...")
        llm.train(train_file, val_file, args.output_dir, batch_size=args.batch_size, max_steps=args.steps)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: {args.output_dir}")
        print("\nTo use:")
        print("  export USE_FINETUNED_LLM=true")
        print(f"  export FINETUNED_MODEL_PATH={args.output_dir}")
        print("  python main.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTry:")
        print("  --batch-size 2 --lora-r 8")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
