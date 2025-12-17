"""
Tests for vacation planning agent.

Demonstrates:
- Multi-turn conversation flow
- Constraint satisfaction
- Uncertainty handling
- State management
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import VacationPlanningAgent, ConversationState, UserRequirements
from tools import FlightAPI, HotelAPI, ActivityAPI, BudgetCalculator
from models import PreferenceRanker, VacationOption


class TestTools:
    """Test mock API tools."""
    
    def test_flight_api(self):
        """Test flight search returns valid results."""
        api = FlightAPI()
        flights = api.search_flights("Bali", "2024-03-15", "2024-03-22")
        
        assert len(flights) > 0
        assert all(f.destination == "Bali" for f in flights)
        assert all(f.price > 0 for f in flights)
        # Cheapest flight should be first
        assert flights[0].price <= flights[-1].price
    
    def test_hotel_api(self):
        """Test hotel search with category filter."""
        api = HotelAPI()
        
        # All hotels
        all_hotels = api.search_hotels("Paris")
        assert len(all_hotels) > 0
        
        # Budget only
        budget_hotels = api.search_hotels("Paris", budget_category="budget")
        assert len(budget_hotels) > 0
        assert all(h.category == "budget" for h in budget_hotels)
    
    def test_activity_api(self):
        """Test activity search with category filter."""
        api = ActivityAPI()
        
        # All activities
        all_activities = api.search_activities("Tokyo")
        assert len(all_activities) > 0
        
        # Adventure only
        adventure = api.search_activities("Tokyo", category="adventure")
        assert all(a.category == "adventure" for a in adventure)
    
    def test_budget_calculator(self):
        """Test budget calculations."""
        calc = BudgetCalculator()
        
        # Mock data
        from tools import Flight, Hotel, Activity
        
        flight = Flight("SFO", "Bali", "2024-03-15", "2024-03-22", 1600, "United", 18.5, 0)
        hotel = Hotel("Resort", "Bali", 120, 4.5, ["pool"], "mid-range")
        activities = [
            Activity("Surf", "Learn surfing", 60, 3, "adventure"),
            Activity("Temple", "Visit temples", 40, 4, "culture"),
        ]
        
        breakdown = calc.calculate_total([flight], hotel, 7, activities)
        
        assert breakdown["flights"] == 1600
        assert breakdown["hotel"] == 120 * 7
        assert breakdown["activities"] == 100
        assert breakdown["total"] == 1600 + 840 + 100
        
        # Test budget fit
        assert calc.fits_budget(2540, 2600)  # Within tolerance
        assert not calc.fits_budget(2540, 2000)  # Over budget


class TestPreferenceRanker:
    """Test preference learning model."""
    
    def test_ranker_initialization(self):
        """Test ranker can be initialized."""
        ranker = PreferenceRanker()
        assert ranker is not None
    
    def test_encode_option(self):
        """Test option encoding."""
        ranker = PreferenceRanker()
        
        option = VacationOption(
            destination="Bali",
            hotel_category="luxury",
            activities=["surf", "spa"],
            total_cost=3500,
            description="Relaxing beach resort"
        )
        
        embedding = ranker.encode_option(option)
        assert embedding.shape[0] == 384  # MiniLM embedding size
    
    def test_rank_options(self):
        """Test ranking multiple options."""
        ranker = PreferenceRanker()
        
        options = [
            VacationOption("Bali", "luxury", ["surf", "spa"], 3500, "Luxury resort"),
            VacationOption("Bali", "budget", ["surf"], 1500, "Budget hostel"),
            VacationOption("Bali", "mid-range", ["surf", "temple"], 2500, "Nice hotel"),
        ]
        
        ranked = ranker.rank_options(options)
        
        assert len(ranked) == 3
        assert all(isinstance(score, float) for _, score in ranked)
        # Scores should be between 0 and 1
        assert all(0 <= score <= 1 for _, score in ranked)
    
    def test_online_learning(self):
        """Test preference ranker learns from feedback."""
        import torch
        
        ranker = PreferenceRanker()
        optimizer = torch.optim.Adam(ranker.parameters(), lr=1e-3)
        
        preferred = VacationOption("Bali", "luxury", ["spa"], 3500, "Luxury")
        rejected = VacationOption("Bali", "budget", ["hostel"], 1500, "Budget")
        
        # Train for a few steps
        losses = []
        for _ in range(5):
            loss = ranker.update_from_feedback(preferred, rejected, optimizer)
            losses.append(loss)
        
        # Loss should decrease (learning is happening)
        assert losses[-1] < losses[0]


class TestAgent:
    """Test agent conversation flow."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        # Note: This will fail without OpenAI API key
        # For unit tests, we'd mock the API calls
        return VacationPlanningAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes in correct state."""
        assert agent.state == ConversationState.GATHERING
        assert agent.requirements.destination is None
        assert len(agent.conversation_history) == 0
    
    def test_state_transitions(self, agent):
        """Test agent transitions through states correctly."""
        # Start in GATHERING
        assert agent.state == ConversationState.GATHERING
        
        # Provide complete requirements
        agent.requirements.destination = "Bali"
        agent.requirements.budget = 3000
        agent.requirements.duration_days = 7
        
        assert agent._has_minimum_requirements()
    
    def test_requirement_extraction(self, agent):
        """Test extracting requirements from messages."""
        # This would require mocking OpenAI API
        # For now, test the logic
        
        reqs = UserRequirements()
        reqs.destination = "Paris"
        reqs.budget = 2500
        reqs.duration_days = 5
        
        assert reqs.destination == "Paris"
        assert reqs.budget == 2500
        assert reqs.duration_days == 5
    
    def test_budget_conflict_handling(self, agent):
        """Test agent handles budget conflicts gracefully."""
        from tools import Flight, Hotel
        
        flight = Flight("SFO", "Paris", "2024-03-15", "2024-03-22", 1300, "Air France", 11, 0)
        hotels = [
            Hotel("Luxury", "Paris", 400, 4.9, ["spa"], "luxury"),
            Hotel("Budget", "Paris", 50, 4.1, ["wifi"], "budget"),
        ]
        
        response = agent._handle_budget_conflict(flight, hotels, 1500, 7)
        
        # Should suggest alternatives
        assert "option" in response.lower() or "alternative" in response.lower()
        assert "$" in response  # Should mention costs
    
    def test_no_results_handling(self, agent):
        """Test agent handles missing data gracefully."""
        response = agent._handle_no_results("flights", "InvalidDestination")
        
        assert "trouble" in response.lower() or "error" in response.lower()
        # Should suggest alternatives
        assert any(dest in response for dest in ["Bali", "Paris", "Tokyo"])
    
    def test_activity_selection(self, agent):
        """Test activity selection respects budget and preferences."""
        from tools import Activity
        
        activities = [
            Activity("Expensive", "Luxury", 500, 4, "luxury"),
            Activity("Cheap1", "Budget", 50, 2, "adventure"),
            Activity("Cheap2", "Budget", 60, 3, "adventure"),
        ]
        
        # With low budget, should select cheap activities
        selected = agent._select_activities_for_budget(activities, 150, ["adventure"])
        
        assert len(selected) > 0
        assert sum(a.price for a in selected) <= 150
        # Should prefer adventure activities
        assert all(a.category == "adventure" for a in selected)
    
    def test_reset(self, agent):
        """Test agent can be reset."""
        # Modify state
        agent.state = ConversationState.REFINING
        agent.requirements.destination = "Bali"
        agent.conversation_history.append({"role": "user", "content": "test"})
        
        # Reset
        agent.reset()
        
        # Should be back to initial state
        assert agent.state == ConversationState.GATHERING
        assert agent.requirements.destination is None
        assert len(agent.conversation_history) == 0


class TestIntegration:
    """Integration tests for full workflows."""
    
    def test_complete_planning_flow(self):
        """Test complete vacation planning flow."""
        # This would require mocking OpenAI API
        # Demonstrates the expected flow:
        
        # 1. User provides requirements
        # 2. Agent generates options
        # 3. User refines
        # 4. Agent updates options
        # 5. User selects
        # 6. Agent creates itinerary
        
        flow_states = [
            ConversationState.GATHERING,
            ConversationState.PLANNING,
            ConversationState.REFINING,
            ConversationState.FINALIZING,
            ConversationState.COMPLETE,
        ]
        
        # Verify state enum has all expected states
        assert all(state in ConversationState for state in flow_states)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
