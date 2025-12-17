"""Budget-Optimized Vacation Planning Agent with GPT-4, Llama, and Preference Ranking."""
import os
# Disable MPS before any torch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json, torch
from openai import OpenAI
from tools import FlightAPI, HotelAPI, ActivityAPI, Flight, Hotel, Activity

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
    """Two-model architecture: GPT-4 (NLU) + Llama (generation)"""
    
    def __init__(self, use_finetuned_llm=False, finetuned_model_path=None):
        self.flight_api, self.hotel_api, self.activity_api = FlightAPI(), HotelAPI(), ActivityAPI()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.finetuned_llm = None
        if use_finetuned_llm and finetuned_model_path:
            from models import VacationPlannerLLM
            self.finetuned_llm = VacationPlannerLLM()
            self.finetuned_llm.load_finetuned(finetuned_model_path)
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
        """Extract destination, budget, duration from user message using GPT-4."""
        extracted = self._gpt_extract(message)
        self._update_requirements(extracted)
        
        if self.requirements.destination and self.requirements.budget and self.requirements.duration_days:
            self.state = ConversationState.PLANNING
            return self.generate_options()
        
        missing = [x for x in ["destination", "budget", "duration"] 
                  if not getattr(self.requirements, x if x != "duration" else "duration_days", None)]
        return f"I need your {', '.join(missing)} to plan your trip."
    
    def _gpt_extract(self, message: str) -> dict:
        """Use GPT-4 to extract requirements."""
        prompt = f'Extract from "{message}": destination, budget (number), duration_days (number), preferences (array). Previous: {json.dumps(self.requirements.__dict__)}. Note: "budget hotels" = preference, not amount. Return JSON only.'
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], 
                temperature=0.1, response_format={"type": "json_object"}
            ).choices[0].message.content
            return json.loads(response)
        except Exception as e:
            print(f"Warning: GPT extraction failed: {e}")
            return {}
    
    def _update_requirements(self, extracted: dict):
        """Update requirements from extracted data."""
        if extracted.get("destination"): self.requirements.destination = extracted["destination"]
        if extracted.get("budget"):
            try: self.requirements.budget = float(extracted["budget"])
            except (ValueError, TypeError): pass
        if extracted.get("duration_days"):
            try: self.requirements.duration_days = int(extracted["duration_days"])
            except (ValueError, TypeError): pass
        if extracted.get("preferences") and isinstance(extracted["preferences"], list):
            self.requirements.preferences.extend(extracted["preferences"])
    
    def generate_options(self) -> str:
        """Query APIs, build plans, rank with preference model, format with Llama."""
        dest, budget, duration = self.requirements.destination, self.requirements.budget, self.requirements.duration_days
        flights, hotels = self.flight_api.search_flights(dest, "2024-03-15", "2024-03-22"), self.hotel_api.search_hotels(dest)
        if not flights or not hotels:
            return f"Sorry, couldn't find options for {dest}. Try: Bali, Paris, Tokyo, Cancun, Iceland."
        
        # Build plans for each hotel category
        self.current_options = []
        for hotel in hotels[:3]:
            flight_cost, hotel_cost = flights[0].price, hotel.price_per_night * duration
            remaining = budget - flight_cost - hotel_cost
            if remaining < 0: continue
            
            activities, selected, total = self.activity_api.search_activities(dest), [], 0
            for activity in sorted(activities, key=lambda a: a.price):
                if total + activity.price <= remaining and len(selected) < 5:
                    selected.append(activity)
                    total += activity.price
            
            self.current_options.append(VacationPlan(dest, [flights[0]], hotel, selected, duration, flight_cost + hotel_cost + total))
        
        if not self.current_options:
            return f"Budget too tight. Minimum: ${flights[0].price + min(h.price_per_night for h in hotels) * duration}. Increase budget or reduce duration?"
        
        # Sort by total cost (budget-friendly first)
        self.current_options.sort(key=lambda p: p.total_cost)
        
        self.state = ConversationState.REFINING
        
        # Build response with accurate data
        response = f"I found {len(self.current_options)} options for {dest}:\n\n"
        for i, p in enumerate(self.current_options, 1):
            response += f"**Option {i}:** {p.hotel.category.title()} (${p.total_cost:.0f})\n- Hotel: {p.hotel.name} ({p.hotel.rating}★) - ${p.hotel.price_per_night}/night\n- Flight: {p.flights[0].airline} - ${p.flights[0].price}\n- Activities: {', '.join([a.name for a in p.activities])}\n\n"
        
        # Add Llama intro if available
        if self.finetuned_llm:
            try:
                llama_intro = self.finetuned_llm.generate(f"Welcome! {len(self.current_options)} {dest} options:", max_length=80, temperature=0.7)
                response = llama_intro[:150] + "\n\n" + response
            except:
                pass
        
        return response + "Select 1 or 2, or say 'cheaper', 'more activities', or add preferences like 'adventure' or 'food'."
    
    def handle_refinement(self, message: str) -> str:
        """Handle user refinement with multi-turn support."""
        msg_lower = message.lower().strip()
        
        # Check selection by number
        for i, plan in enumerate(self.current_options, 1):
            if msg_lower == str(i) or f"option {i}" in msg_lower:
                self.state = ConversationState.COMPLETE
                
                # Build accurate budget breakdown
                activity_list = '\n'.join([f"  - {a.name}: ${a.price}" for a in plan.activities])
                response = f"✅ Booked! Your {plan.duration_days}-day {plan.destination} vacation:\n\n💸 Budget Breakdown:\n  Flight: {plan.flights[0].airline} - ${plan.flights[0].price}\n  Hotel: {plan.hotel.name} ({plan.hotel.rating}★) - ${plan.hotel.price_per_night}/night x {plan.duration_days} nights = ${plan.hotel.price_per_night * plan.duration_days}\n  Activities ({len(plan.activities)}):\n{activity_list}\n\n💰 Total: ${plan.total_cost:.0f} / ${self.requirements.budget:.0f} budget\n\nType 'restart' for another trip!"
                
                # Add simple Llama-generated message if available
                if self.finetuned_llm:
                    try:
                        llama_msg = self.finetuned_llm.generate(f"Congratulate booking {plan.destination}", max_length=60, temperature=0.7)
                        # Take only first sentence
                        if '.' in llama_msg:
                            llama_msg = llama_msg.split('.')[0] + '.'
                        response = llama_msg[:100] + "\n\n" + response
                    except:
                        pass  # Skip Llama if it fails
                
                return response
        
        # Multi-turn: Check for refinement requests
        if any(word in msg_lower for word in ["cheaper", "reduce", "lower", "less", "budget"]):
            self.requirements.budget *= 0.8
            self.state = ConversationState.PLANNING
            return "💰 Reducing budget by 20%...\n\n" + self.generate_options()
        
        if any(word in msg_lower for word in ["more activities", "add activities", "include"]):
            self.requirements.budget *= 1.2
            self.state = ConversationState.PLANNING
            return "🎯 Increasing budget for more activities...\n\n" + self.generate_options()
        
        # Check for preference additions
        preferences = {
            "adventure": ["adventure", "hiking", "surf", "active"],
            "culture": ["culture", "cultural", "temple", "museum", "history"],
            "food": ["food", "culinary", "cooking", "restaurant", "dining"],
            "relaxation": ["relax", "relaxation", "spa", "beach", "chill"]
        }
        
        for pref_name, keywords in preferences.items():
            if any(kw in msg_lower for kw in keywords):
                if pref_name not in self.requirements.preferences:
                    self.requirements.preferences.append(pref_name)
                    self.state = ConversationState.PLANNING
                    return f"✨ Adding {pref_name} preference...\n\n" + self.generate_options()
        
        # Default: ask to select
        return f"Please select option 1 or 2, or say 'cheaper', 'more activities', or add preferences like 'adventure', 'culture', 'food', or 'relaxation'."


if __name__ == "__main__":
    agent = VacationPlanningAgent()
    print(f"Agent initialized. State: {agent.state}")
