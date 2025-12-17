"""Generate synthetic vacation planning conversations for training."""

import json
import random
from typing import List, Dict


class VacationDataGenerator:
    """Generate synthetic training data."""
    
    DESTINATIONS = ["Bali", "Cancun", "Paris", "Tokyo", "Iceland"]
    BUDGETS = [1500, 2000, 2500, 3000, 3500, 4000, 5000]
    DURATIONS = [3, 5, 7, 10, 14]
    PREFERENCES = ["adventure", "relaxation", "culture", "food", "luxury", "budget-friendly"]
    
    TEMPLATES = [
        # Template 1: Budget-conscious
        [
            ("user", "I want a {duration}-day trip to {destination} with ${budget} budget"),
            ("assistant", "Great! For {destination} with ${budget}, I found:\n- Flights: ${flight_cost}\n- Hotels: ${hotel_low}-${hotel_high}/night\n- Remaining: ${remaining} for activities\n\nPrefer budget, mid-range, or luxury?"),
            ("user", "{hotel_pref}"),
            ("assistant", "Perfect! {hotel_name} at ${hotel_price}/night. Activities:\n1. {activity1} (${act1_price})\n2. {activity2} (${act2_price})\n\nTotal: ${total}. Good?"),
        ],
        # Template 2: Preference-driven
        [
            ("user", "I want a {preference} vacation to {destination}"),
            ("assistant", "{destination} is perfect for {preference}! Budget and duration?"),
            ("user", "${budget} for {duration} days"),
            ("assistant", "With ${budget} for {duration} days:\n\nFlights: ${flight_cost}\nHotel: {hotel_name} - ${hotel_price}/night\nActivities: {activity1}, {activity2}\n\nTotal: ${total}"),
        ],
        # Template 3: Budget negotiation
        [
            ("user", "I want luxury {destination} for ${budget}"),
            ("assistant", "${budget} is tight for luxury. Options:\nA) Luxury hotel, {short_duration} days\nB) Mid-range hotel, {duration} days\nC) Different destination\n\nWhich?"),
            ("user", "Option B"),
            ("assistant", "Smart! {duration}-day itinerary:\n\nFlights: ${flight_cost}\nHotel: {hotel_name} - ${hotel_price}/night\nActivities: {activity1}, {activity2}\n\nTotal: ${total}"),
        ],
    ]
    
    def generate_conversation(self) -> Dict:
        """Generate one conversation."""
        template = random.choice(self.TEMPLATES)
        
        # Random parameters
        destination = random.choice(self.DESTINATIONS)
        budget = random.choice(self.BUDGETS)
        duration = random.choice(self.DURATIONS)
        preference = random.choice(self.PREFERENCES)
        
        # Realistic costs
        flight_cost = self._estimate_flight(destination)
        hotel_price = self._estimate_hotel(destination)
        hotel_low = int(hotel_price * 0.4)
        hotel_high = int(hotel_price * 2)
        
        activities = self._generate_activities(destination, 2)
        activity_cost = sum(a["price"] for a in activities)
        
        total = flight_cost + (hotel_price * duration) + activity_cost
        remaining = budget - flight_cost - (hotel_price * duration)
        
        # Fill template
        params = {
            "destination": destination,
            "budget": budget,
            "duration": duration,
            "preference": preference,
            "flight_cost": flight_cost,
            "hotel_price": hotel_price,
            "hotel_low": hotel_low,
            "hotel_high": hotel_high,
            "hotel_name": f"{destination} Resort",
            "hotel_pref": random.choice(["Mid-range", "Budget", "Luxury"]),
            "activity1": activities[0]["name"],
            "activity2": activities[1]["name"],
            "act1_price": activities[0]["price"],
            "act2_price": activities[1]["price"],
            "total": int(total),
            "remaining": int(remaining),
            "short_duration": duration - 2,
        }
        
        messages = []
        for role, content in template:
            messages.append({"role": role, "content": content.format(**params)})
        
        return {"messages": messages, "metadata": {"destination": destination, "budget": budget}}
    
    def _estimate_flight(self, destination: str) -> int:
        costs = {"Bali": 1600, "Cancun": 900, "Paris": 1300, "Tokyo": 1800, "Iceland": 1100}
        return costs.get(destination, 1000) + random.randint(-100, 100)
    
    def _estimate_hotel(self, destination: str) -> int:
        costs = {"Bali": 120, "Cancun": 150, "Paris": 180, "Tokyo": 140, "Iceland": 130}
        return costs.get(destination, 120)
    
    def _generate_activities(self, destination: str, count: int) -> List[Dict]:
        pools = {
            "Bali": [("Surf Lesson", 60), ("Temple Tour", 40), ("Cooking Class", 70)],
            "Cancun": [("Snorkeling", 80), ("Mayan Ruins", 120), ("Taco Tour", 50)],
            "Paris": [("Eiffel Tower", 30), ("Louvre Tour", 60), ("Cooking Class", 100)],
            "Tokyo": [("Sushi Making", 120), ("Temple Tour", 40), ("Mt. Fuji", 150)],
            "Iceland": [("Northern Lights", 100), ("Blue Lagoon", 80), ("Glacier Hike", 150)],
        }
        activities = pools.get(destination, [("Activity", 50)])
        selected = random.sample(activities, min(count, len(activities)))
        return [{"name": name, "price": price} for name, price in selected]
    
    def generate_dataset(self, num_examples: int = 150) -> List[Dict]:
        """Generate full dataset."""
        return [self.generate_conversation() for _ in range(num_examples)]
    
    def save_dataset(self, filepath: str, num_examples: int = 150):
        """Generate and save dataset."""
        dataset = self.generate_dataset(num_examples)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Generated {len(dataset)} conversations → {filepath}")
        return dataset


if __name__ == "__main__":
    generator = VacationDataGenerator()
    generator.save_dataset("vacation_train.json", 150)
    generator.save_dataset("vacation_val.json", 30)
    print("\nDataset ready for training!")
