"""
Simple test script for vacation planning agent.
Tests the core functionality without requiring API keys.
"""

import os
os.environ["OPENAI_API_KEY"] = "test-key"  # Mock key for testing

from agent import VacationPlanningAgent, ConversationState
from tools import FlightAPI, HotelAPI, ActivityAPI


def test_tools():
    """Test that mock APIs return data."""
    print("Testing mock APIs...")
    
    flight_api = FlightAPI()
    flights = flight_api.search_flights("Bali", "2024-03-15", "2024-03-22")
    assert len(flights) > 0, "Flight API should return results"
    assert flights[0].destination == "Bali", "Flight destination should match"
    print(f"✅ Flight API: Found {len(flights)} flights to Bali")
    
    hotel_api = HotelAPI()
    hotels = hotel_api.search_hotels("Paris")
    assert len(hotels) > 0, "Hotel API should return results"
    print(f"✅ Hotel API: Found {len(hotels)} hotels in Paris")
    
    activity_api = ActivityAPI()
    activities = activity_api.search_activities("Tokyo")
    assert len(activities) > 0, "Activity API should return results"
    print(f"✅ Activity API: Found {len(activities)} activities in Tokyo")


def test_agent_initialization():
    """Test agent initializes correctly."""
    print("\nTesting agent initialization...")
    
    agent = VacationPlanningAgent()
    assert agent.state == ConversationState.GATHERING, "Agent should start in GATHERING state"
    assert agent.requirements.destination is None, "Initial destination should be None"
    assert agent.requirements.budget is None, "Initial budget should be None"
    print("✅ Agent initialized successfully")


def test_option_generation():
    """Test that agent generates vacation options."""
    print("\nTesting option generation...")
    
    agent = VacationPlanningAgent()
    
    # Manually set requirements (skip GPT-4 extraction)
    agent.requirements.destination = "Bali"
    agent.requirements.budget = 3000
    agent.requirements.duration_days = 7
    agent.state = ConversationState.PLANNING
    
    # Generate options
    response = agent.generate_options()
    
    assert "Bali" in response, "Response should mention destination"
    assert "Option" in response, "Response should present options"
    assert len(agent.current_options) > 0, "Should generate at least one option"
    print(f"✅ Generated {len(agent.current_options)} vacation options")
    
    # Check budget constraints
    for option in agent.current_options:
        assert option.total_cost <= 3000, f"Option cost ${option.total_cost} exceeds budget"
    print("✅ All options within budget")


def test_multi_turn_flow():
    """Test multi-turn conversation flow."""
    print("\nTesting multi-turn flow...")
    
    agent = VacationPlanningAgent()
    
    # Set up initial state with options
    agent.requirements.destination = "Tokyo"
    agent.requirements.budget = 2500
    agent.requirements.duration_days = 5
    agent.state = ConversationState.PLANNING
    agent.generate_options()
    
    initial_budget = agent.requirements.budget
    initial_count = len(agent.current_options)
    
    # Test "cheaper" refinement
    agent.state = ConversationState.REFINING
    response = agent.handle_refinement("make it cheaper")
    
    assert agent.requirements.budget < initial_budget, "Budget should decrease"
    print(f"✅ Budget reduced from ${initial_budget} to ${agent.requirements.budget}")
    
    # Test "more activities" refinement
    agent.handle_refinement("more activities")
    assert agent.requirements.budget > initial_budget * 0.8, "Budget should increase"
    print("✅ Multi-turn refinement working")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("🧪 VACATION PLANNING AGENT - TESTS")
    print("="*60)
    
    try:
        test_tools()
        test_agent_initialization()
        test_option_generation()
        test_multi_turn_flow()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
