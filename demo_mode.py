"""
Demo mode utilities for running agent without API key.
Simple rule-based NLU for testing and demos.
"""

import re
from typing import Dict, List


def simple_extract(text: str) -> Dict:
    """
    Simple rule-based extraction for demo mode.
    Extracts destination, budget, duration, and preferences from text.
    """
    extracted = {}
    text_lower = text.lower()
    
    # Extract budget
    budget_match = re.search(r'\$(\d+(?:,\d+)?)', text)
    if budget_match:
        extracted["budget"] = int(budget_match.group(1).replace(',', ''))
    
    # Extract duration
    duration_match = re.search(r'(\d+)\s*days?', text_lower)
    if duration_match:
        extracted["duration_days"] = int(duration_match.group(1))
    
    # Extract destination (hardcoded list)
    destinations = ["Bali", "Paris", "Tokyo", "Cancun", "Iceland"]
    for dest in destinations:
        if dest.lower() in text_lower:
            extracted["destination"] = dest
            break
    
    # Extract preferences
    preferences = []
    if "adventure" in text_lower:
        preferences.append("adventure")
    if "relax" in text_lower or "relaxation" in text_lower:
        preferences.append("relaxation")
    if "culture" in text_lower or "cultural" in text_lower:
        preferences.append("culture")
    if "food" in text_lower or "culinary" in text_lower:
        preferences.append("food")
    if "luxury" in text_lower:
        preferences.append("luxury")
    if "budget" in text_lower and "friendly" in text_lower:
        preferences.append("budget-friendly")
    
    if preferences:
        extracted["preferences"] = preferences
    
    return extracted


def simple_intent_classification(message: str) -> str:
    """
    Simple intent classification for demo mode.
    Returns: 'reduce_cost', 'add_features', 'select_option', or 'clarify'
    """
    message_lower = message.lower()
    
    # Check for cost reduction
    if any(word in message_lower for word in ["cheap", "reduce", "lower", "less", "budget"]):
        return "reduce_cost"
    
    # Check for adding features
    if any(word in message_lower for word in ["more", "add", "include"]):
        return "add_features"
    
    # Check for selection
    if "option" in message_lower or any(f"#{i}" in message_lower for i in range(1, 10)):
        return "select_option"
    
    # Default: ask for clarification
    return "clarify"
