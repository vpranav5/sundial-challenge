"""Budget-Optimized Vacation Planning Agent with GPT-4, Llama, and Preference Ranking."""
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json, os, torch
from openai import OpenAI
from tools import FlightAPI, HotelAPI, ActivityAPI, Flight, Hotel, Activity
from models import PreferenceRanker, VacationOption
from demo_mode import simple_extract, simple_intent_classification

class ConversationState(Enum):
    GATHERING, PLANNING, REFINING, COMPLETE = "gathering", "planning", "refining", "complete"

@dataclass
class UserRequirements:
    destination: Optional[str] = None; budget: Optional[float] = None
    duration_days: Optional[int] = None; preferences: List[str] = field(default_factory=list)

@dataclass
class VacationPlan:
    destination: str; flights: List[Flight]; hotel: Hotel; activities: List[Activity]
    duration_days: int; total_cost: float


class VacationPlanningAgent:
    """Three-model architecture: GPT-4 (NLU), Llama (generation), Preference Ranker (personalization)"""
    
    def __init__(self, demo_mode=False, use_finetuned_llm=False, finetuned_model_path=None):
        self.flight_api, self.hotel_api, self.activity_api = FlightAPI(), HotelAPI(), ActivityAPI()
        self.demo_mode = demo_mode
        self.client = None if demo_mode else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.finetuned_llm = None
        if use_finetuned_llm and finetuned_model_path:
            from models import VacationPlannerLLM
            self.finetuned_llm = VacationPlannerLLM()
            self.finetuned_llm.load_finetuned(finetuned_model_path)
        self.ranker = PreferenceRanker()
        self.state = ConversationState.GATHERING
        self.requirements = UserRequirements()
        self.current_options, self.conversation_history = [], []
    
    def process_message(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})
        response = (self.extract_requirements(user_message) if self.state == ConversationState.GATHERING
                   else self.generate_options() if self.state == ConversationState.PLANNING
                   else self.handle_refinement(user_message) if self.state == ConversationState.REFINING
                   else "Your vacation is planned! Type 'restart' for a new trip.")
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    def extract_requirements(self, message: str) -> str:
        """Extract destination, budget, duration from user message using GPT-4 or demo mode."""
        # Use GPT-4 for NLU (or simple regex in demo mode)
        if self.demo_mode:
            extracted = simple_extract(message)
        else:
            prompt = f'Extract vacation requirements from: "{message}"\nReturn JSON with: destination, budget, duration_days, preferences\nPrevious: {json.dumps(self.requirements.__dict__)}\nReturn valid JSON only.'
            extracted = json.loads(self.client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1
            ).choices[0].message.content) if self.client else {}
        
        # Update requirements
        if extracted.get("destination"): self.requirements.destination = extracted["destination"]
        if extracted.get("budget"): self.requirements.budget = float(extracted["budget"])
        if extracted.get("duration_days"): self.requirements.duration_days = int(extracted["duration_days"])
        if extracted.get("preferences"): self.requirements.preferences.extend(extracted["preferences"])
        
        # If complete, move to planning
        if self.requirements.destination and self.requirements.budget and self.requirements.duration_days:
            self.state = ConversationState.PLANNING
            return self.generate_options()
        
        # Ask for missing info
        missing = [x for x in ["destination", "budget", "duration"] 
                  if not getattr(self.requirements, x if x != "duration" else "duration_days", None)]
        return f"I need your {', '.join(missing)} to plan your trip."
    
    def generate_options(self) -> str:
        """Query APIs, build plans, rank with preference model, format with Llama."""
        # Get user requirements
        dest, budget, duration = self.requirements.destination, self.requirements.budget, self.requirements.duration_days
        
        # Query flight and hotel APIs
        flights = self.flight_api.search_flights(dest, "2024-03-15", "2024-03-22")
        hotels = self.hotel_api.search_hotels(dest)
        if not flights or not hotels:
            return f"Sorry, couldn't find options for {dest}. Try: Bali, Paris, Tokyo, Cancun, Iceland."
        
        # Build vacation plans for each hotel category
        self.current_options = []
        for hotel in hotels[:3]:  # Try luxury, mid-range, budget
            flight_cost = flights[0].price
            hotel_cost = hotel.price_per_night * duration
            remaining = budget - flight_cost - hotel_cost
            if remaining < 0: continue  # Skip if over budget
            
            # Select activities within remaining budget
            activities = self.activity_api.search_activities(dest)
            selected, total = [], 0
            for activity in sorted(activities, key=lambda a: a.price):
                if total + activity.price <= remaining and len(selected) < 5:
                    selected.append(activity)
                    total += activity.price
            
            # Create plan
            plan = VacationPlan(dest, [flights[0]], hotel, selected, duration, flight_cost + hotel_cost + total)
            self.current_options.append(plan)
        
        # Handle budget too low
        if not self.current_options:
            min_cost = flights[0].price + min(h.price_per_night for h in hotels) * duration
            return f"Budget too tight. Minimum: ${min_cost}. Increase budget or reduce duration?"
        
        # Rank options using preference model
        vacation_options = [VacationOption(p.destination, p.hotel.category, [a.name for a in p.activities], 
                                          p.total_cost, f"{len(p.activities)} activities") 
                           for p in self.current_options]
        ranked = self.ranker.rank_options(vacation_options)
        self.current_options = [next(p for p in self.current_options if p.hotel.category == vo.hotel_category) 
                               for vo, _ in ranked]
        
        # Generate response with Llama or template
        if self.finetuned_llm:
            self.state = ConversationState.REFINING
            return self.finetuned_llm.generate(f"Present {len(self.current_options)} vacation options for {dest}", max_length=200)
        
        # Template fallback
        response = f"I found {len(self.current_options)} options for {dest}:\n\n"
        for i, p in enumerate(self.current_options, 1):
            response += f"**Option {i}:** {p.hotel.category.title()} (${p.total_cost:.0f})\n"
            response += f"- Hotel: {p.hotel.name} ({p.hotel.rating}★) - ${p.hotel.price_per_night}/night\n"
            response += f"- Flight: {p.flights[0].airline} - ${p.flights[0].price}\n"
            response += f"- Activities: {', '.join([a.name for a in p.activities])}\n\n"
        response += "Which option? Or ask me to adjust (cheaper, more activities, etc.)"
        self.state = ConversationState.REFINING
        return response
    
    def handle_refinement(self, message: str) -> str:
        """Handle user refinement - selection or intent classification with GPT-4."""
        msg_lower = message.lower()
        
        # Check if user selected an option
        for i, plan in enumerate(self.current_options, 1):
            if f"option {i}" in msg_lower or f"#{i}" in msg_lower:
                self.state = ConversationState.COMPLETE
                
                # Generate detailed itinerary with Llama or template
                if self.finetuned_llm:
                    return self.finetuned_llm.generate(
                        f"Create detailed itinerary: {plan.destination}, {plan.duration_days} days, ${plan.total_cost}", 
                        max_length=300)
                
                # Template fallback
                return (f"Perfect! Your {plan.duration_days}-day {plan.destination} trip:\n\n"
                       f"Flight: {plan.flights[0].airline} - ${plan.flights[0].price}\n"
                       f"Hotel: {plan.hotel.name} ({plan.hotel.rating}★) - ${plan.hotel.price_per_night}/night\n"
                       f"Activities: {', '.join([a.name for a in plan.activities])}\n\n"
                       f"Total: ${plan.total_cost} | Budget: ${self.requirements.budget}\n\n"
                       f"Confirm to finalize!")
        
        # Classify intent with GPT-4 (or simple rules in demo mode)
        if self.demo_mode:
            intent = simple_intent_classification(message)
        else:
            prompt = f'User said: "{message}"\nWhat do they want? A) Select option B) Cheaper C) Add features D) Different options\nReply with letter only.'
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1
            ).choices[0].message.content.strip().upper() if self.client else "E"
            intent_map = {"A": "select", "B": "reduce_cost", "C": "add_features", "D": "alternatives"}
            intent = intent_map.get(response, "clarify")
        
        # Handle intent
        if intent == "reduce_cost":
            return "I can reduce costs by: 1) Budget hotels 2) Fewer activities 3) Shorter trip. Which?"
        elif intent == "add_features":
            return "What would you like to add? (adventure, culture, food, relaxation)"
        else:
            return "Would you like to: 1) Select an option 2) Make it cheaper 3) Add activities?"
    
    def reset(self):
        self.state, self.requirements, self.current_options, self.conversation_history = ConversationState.GATHERING, UserRequirements(), [], []


if __name__ == "__main__":
    agent = VacationPlanningAgent()
    print(f"Agent initialized. State: {agent.state}")
