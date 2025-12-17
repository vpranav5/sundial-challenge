"""Mock API tools for vacation planning."""

from typing import List
from dataclasses import dataclass


@dataclass
class Flight:
    origin: str
    destination: str
    departure_date: str
    return_date: str
    price: float
    airline: str
    duration_hours: float
    stops: int


@dataclass
class Hotel:
    name: str
    location: str
    price_per_night: float
    rating: float
    amenities: List[str]
    category: str  # budget, mid-range, luxury


@dataclass
class Activity:
    name: str
    description: str
    price: float
    duration_hours: float
    category: str  # adventure, culture, relaxation, food


class FlightAPI:
    """Mock flight search API."""
    
    DESTINATIONS = {
        "bali": ("Singapore Airlines", 1600, 18.5),
        "cancun": ("Delta", 900, 5.5),
        "paris": ("Air France", 1300, 11.0),
        "tokyo": ("ANA", 1800, 13.0),
        "iceland": ("Icelandair", 1100, 8.0)
    }
    
    def search_flights(self, destination: str, departure_date: str, return_date: str, origin: str = "SFO") -> List[Flight]:
        dest_lower = destination.lower()
        if dest_lower not in self.DESTINATIONS:
            return []
        
        airline, base_price, duration = self.DESTINATIONS[dest_lower]
        return [Flight(origin, destination, departure_date, return_date, base_price, airline, duration, 0)]


class HotelAPI:
    """Mock hotel search API."""
    
    HOTELS = {
        "bali": [
            ("Luxury Beach Resort", 250, 4.8, ["pool", "spa", "beach"], "luxury"),
            ("Ubud Boutique Hotel", 120, 4.5, ["pool", "yoga"], "mid-range"),
            ("Cozy Guesthouse", 45, 4.2, ["wifi", "breakfast"], "budget"),
        ],
        "cancun": [
            ("All-Inclusive Resort", 300, 4.7, ["all-inclusive", "beach"], "luxury"),
            ("Beachfront Hotel", 150, 4.4, ["pool", "beach"], "mid-range"),
            ("Downtown Hostel", 35, 4.0, ["wifi"], "budget"),
        ],
        "paris": [
            ("5-Star Luxury Hotel", 400, 4.9, ["concierge", "spa"], "luxury"),
            ("Charming Boutique", 180, 4.6, ["breakfast", "wifi"], "mid-range"),
            ("Budget Hostel", 50, 4.1, ["wifi"], "budget"),
        ],
        "tokyo": [
            ("Luxury Shibuya Hotel", 350, 4.8, ["restaurant", "gym"], "luxury"),
            ("Modern Business Hotel", 140, 4.5, ["wifi", "breakfast"], "mid-range"),
            ("Capsule Hotel", 40, 4.3, ["wifi"], "budget"),
        ],
        "iceland": [
            ("Luxury Reykjavik Hotel", 280, 4.7, ["spa", "northern-lights"], "luxury"),
            ("Cozy Guesthouse", 130, 4.5, ["breakfast"], "mid-range"),
            ("Budget Hostel", 45, 4.2, ["wifi"], "budget"),
        ]
    }
    
    def search_hotels(self, destination: str) -> List[Hotel]:
        dest_lower = destination.lower()
        if dest_lower not in self.HOTELS:
            return []
        
        hotels = []
        for name, price, rating, amenities, category in self.HOTELS[dest_lower]:
            hotels.append(Hotel(name, destination, price, rating, amenities, category))
        return hotels


class ActivityAPI:
    """Mock activity/experience API."""
    
    ACTIVITIES = {
        "bali": [
            ("Surf Lesson", "Learn to surf at Kuta Beach", 60, 3, "adventure"),
            ("Temple Tour", "Visit ancient temples in Ubud", 40, 4, "culture"),
            ("Spa Day", "Traditional Balinese massage", 80, 3, "relaxation"),
            ("Cooking Class", "Learn Indonesian cuisine", 70, 4, "food"),
            ("Volcano Hike", "Sunrise hike up Mount Batur", 90, 6, "adventure"),
        ],
        "cancun": [
            ("Snorkeling Tour", "Explore coral reefs", 80, 4, "adventure"),
            ("Mayan Ruins", "Visit Chichen Itza", 120, 8, "culture"),
            ("Beach Day", "Relax at Playa Delfines", 0, 4, "relaxation"),
            ("Taco Tour", "Street food tasting", 50, 3, "food"),
            ("Cenote Diving", "Dive in underground caves", 150, 5, "adventure"),
        ],
        "paris": [
            ("Eiffel Tower", "Skip-the-line tickets", 30, 2, "culture"),
            ("Louvre Tour", "Guided museum tour", 60, 4, "culture"),
            ("Seine Cruise", "Evening river cruise", 40, 2, "relaxation"),
            ("Cooking Class", "French pastry workshop", 100, 4, "food"),
            ("Bike Tour", "Explore Paris by bike", 50, 3, "adventure"),
        ],
        "tokyo": [
            ("Sushi Making", "Learn from master chef", 120, 3, "food"),
            ("Temple Tour", "Visit historic temples", 40, 4, "culture"),
            ("Karaoke Night", "Private karaoke room", 30, 3, "relaxation"),
            ("Mt. Fuji Hike", "Day trip to Mt. Fuji", 150, 10, "adventure"),
            ("Robot Restaurant", "Unique dinner show", 80, 2, "culture"),
        ],
        "iceland": [
            ("Northern Lights", "Aurora hunting tour", 100, 4, "adventure"),
            ("Blue Lagoon", "Geothermal spa", 80, 3, "relaxation"),
            ("Glacier Hike", "Explore ice caves", 150, 6, "adventure"),
            ("Golden Circle", "Geysers and waterfalls", 90, 8, "culture"),
            ("Whale Watching", "Boat tour", 110, 4, "adventure"),
        ]
    }
    
    def search_activities(self, destination: str) -> List[Activity]:
        dest_lower = destination.lower()
        if dest_lower not in self.ACTIVITIES:
            return []
        
        activities = []
        for name, desc, price, duration, category in self.ACTIVITIES[dest_lower]:
            activities.append(Activity(name, desc, price, duration, category))
        return activities


class BudgetCalculator:
    """Calculate vacation budgets."""
    
    @staticmethod
    def calculate_total(flights: List[Flight], hotel: Hotel, num_nights: int, activities: List[Activity]) -> dict:
        flight_cost = sum(f.price for f in flights)
        hotel_cost = hotel.price_per_night * num_nights
        activity_cost = sum(a.price for a in activities)
        return {
            "flights": flight_cost,
            "hotel": hotel_cost,
            "activities": activity_cost,
            "total": flight_cost + hotel_cost + activity_cost
        }
