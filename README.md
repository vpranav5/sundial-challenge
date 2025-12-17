# Budget-Optimized Vacation Planning Agent

An AI agent that plans vacations with budget optimization and adaptive preference learning. Uses a custom state machine (no LangGraph) with GPT-4 for orchestration, a neural preference ranker for personalization, and LoRA fine-tuned Llama 3.2 3B for domain-specific generation. Demonstrates multi-turn conversation, goal-oriented planning, and uncertainty handling through trade-off negotiation.

## Architecture

GPT-4 handles intent understanding and orchestration. A custom-trained neural preference ranker learns user preferences online during conversation. LoRA fine-tuned Llama 3.2 3B generates vacation-specific text (demonstrates large model training, 4-bit quantization, parameter-efficient fine-tuning).

## Setup

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add OpenAI API key
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...
```

## Run

```bash
# With GPT-4 (requires API key)
python main.py

# Demo mode (no API key needed)
python main.py --demo
```

## Train

```bash
# Generate synthetic data
python data_gen.py

# Train LoRA model (1-2 hours)
python train.py --steps 100 --batch-size 4
```

## Test

```bash
pytest tests/ -v
```

## Files

- `agent.py` - State machine and orchestration
- `models.py` - LoRA fine-tuning + preference ranker
- `tools.py` - Mock APIs (flights, hotels, activities)
- `data_gen.py` - Synthetic conversation generation
- `train.py` - Training script
- `main.py` - CLI interface
- `tests/` - Unit tests
