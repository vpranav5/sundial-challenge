# Vacation Planning Agent

AI-powered vacation planning agent with multi-turn conversation, budget optimization, and natural language generation. Uses GPT-4 for NLU and fine-tuned Llama for generation.

## Architecture

**Two-model system:**
- **GPT-4** - Natural language understanding (extracts destination, budget, duration, preferences)
- **Llama (LoRA fine-tuned)** - Natural language generation for vacation options

## Quick Start

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run with GPT-4 only
OPENAI_API_KEY=your-key-here python main.py

# 3. Run with GPT-4 + fine-tuned Llama
OPENAI_API_KEY=your-key-here USE_FINETUNED_LLM=true python main.py
```

## Example Usage

```
👤 You: Plan a 7 day trip to Bali with $3000 budget

🤖 Agent: I found 2 options for Bali:

**Option 1:** Budget ($2255)
- Hotel: Cozy Guesthouse (4.2★) - $45/night
- Flight: Singapore Airlines - $1600
- Activities: Temple Tour, Surf Lesson, Cooking Class, Spa Day, Volcano Hike

**Option 2:** Mid-Range ($2780)
- Hotel: Ubud Boutique Hotel (4.5★) - $120/night
- Flight: Singapore Airlines - $1600
- Activities: Temple Tour, Surf Lesson, Cooking Class, Spa Day, Volcano Hike

Select 1 or 2, or say 'cheaper', 'more activities', or add preferences like 'adventure' or 'food'.

👤 You: 1

🤖 Agent: ✅ Booked! Your 7-day Bali vacation:
...
```

## Training Llama

```bash
# Generate synthetic training data
python data_gen.py

# Train on GPU (Colab recommended)
python train.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --steps 100

# Model saved to: ./models/vacation-planner-lora
```

## Testing

```bash
# Run tests
python test_agent.py
```

## Files

- `main.py` - CLI entry point
- `agent.py` - Conversation agent (~170 lines)
- `models.py` - Llama LoRA fine-tuning
- `tools.py` - Mock APIs (flights, hotels, activities)
- `data_gen.py` - Synthetic data generation
- `train.py` - Training script
- `test_agent.py` - Simple tests
