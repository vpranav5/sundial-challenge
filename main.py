"""CLI interface for vacation planning agent."""

import os
import sys
import argparse
from dotenv import load_dotenv
from agent import VacationPlanningAgent


def main():
    # Load environment
    load_dotenv()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OPENAI_API_KEY found.")
        print("Please set OPENAI_API_KEY environment variable.\n")
        sys.exit(1)
    
    # Print banner
    print("\n" + "="*60)
    print("🌴  VACATION PLANNING AGENT  ✈️")
    print("="*60)
    
    # Initialize agent
    print("🔧 Initializing agent...")
    try:
        # Check for fine-tuned model
        use_finetuned = os.getenv("USE_FINETUNED_LLM", "false").lower() == "true"
        model_path = os.getenv("FINETUNED_MODEL_PATH", "./models/vacation-planner-lora")
        
        if use_finetuned and os.path.exists(model_path):
            print(f"📦 Loading fine-tuned Llama from {model_path}...")
            agent = VacationPlanningAgent(use_finetuned_llm=True, finetuned_model_path=model_path)
            print("✅ Agent ready with fine-tuned Llama!\n")
        else:
            agent = VacationPlanningAgent()
            print("✅ Agent ready!\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Conversation loop
    print("Type 'quit' to exit, 'restart' to start over, 'help' for tips\n")
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("\n✈️  Safe travels! Goodbye!")
                break
            if user_input.lower() == 'restart':
                agent.reset()
                print("\n🔄 Starting fresh!\n")
                continue
            if user_input.lower() == 'help':
                print("\n💡 Tips:")
                print("- Start with: destination, budget, duration")
                print("- Example: 'I want to go to Bali for 7 days with $3000'")
                print("- Available: Bali, Paris, Tokyo, Cancun, Iceland\n")
                continue
            
            # Process message
            print("\n🤖 Agent: ", end="", flush=True)
            response = agent.process_message(user_input)
            print(response + "\n")
        
        except KeyboardInterrupt:
            print("\n\n✈️  Safe travels! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
