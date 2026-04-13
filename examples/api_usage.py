"""
HARPO API Usage Examples

Demonstrates how to use HARPO for evaluation, comparison, and explanation.
"""

import json
from harpo import Evaluator, Comparator, Explainer

# ============================================================================
# Example 1: Score a Single Response
# ============================================================================

def example_score():
    """Score a conversational response"""
    
    evaluator = Evaluator(model_path="path/to/model.pt")
    
    context = "User: What movies do you recommend for someone who likes sci-fi?"
    response = "I'd recommend Inception - it's a mind-bending sci-fi thriller with stunning visuals and a complex plot."
    
    scores = evaluator.score(context, response)
    
    print("Score Result:")
    print(f"  Relevance:    {scores.relevance:.3f}")
    print(f"  Diversity:    {scores.diversity:.3f}")
    print(f"  Satisfaction: {scores.satisfaction:.3f}")
    print(f"  Engagement:   {scores.engagement:.3f}")
    
    # Convert to dictionary for JSON serialization
    result_dict = scores.to_dict()
    print(f"  As JSON: {json.dumps(result_dict, indent=2)}")


# ============================================================================
# Example 2: Batch Score Multiple Responses
# ============================================================================

def example_batch_score():
    """Score multiple responses efficiently"""
    
    evaluator = Evaluator(model_path="path/to/model.pt")
    
    contexts = [
        "User: What movies do you recommend?",
        "User: Tell me about Italian cuisine.",
    ]
    
    responses = [
        "I recommend Inception - it's a mind-bending masterpiece.",
        "Italian cuisine features fresh pasta, olive oil, and regional specialties.",
    ]
    
    results = evaluator.batch_score(contexts, responses)
    
    for i, result in enumerate(results):
        print(f"\nResponse {i+1}:")
        print(f"  Avg Score: {(result.relevance + result.diversity + result.satisfaction + result.engagement) / 4:.3f}")


# ============================================================================
# Example 3: Compare Two Responses
# ============================================================================

def example_compare():
    """Compare two different responses"""
    
    comparator = Comparator(model_path="path/to/model.pt")
    
    context = "User: What movies do you recommend?"
    
    response_a = "I recommend Inception."
    response_b = "I recommend Inception. It's a mind-bending sci-fi thriller with stunning visuals, an innovative plot about dreams within dreams, and features an excellent ensemble cast."
    
    comparison = comparator.compare(context, response_a, response_b)
    
    print("Comparison Result:")
    print(f"  P(A > B): {comparison.preference_prob_a:.3f}")
    print(f"  P(B > A): {comparison.preference_prob_b:.3f}")
    print(f"  Margin:   {comparison.margin:.3f}")
    
    if comparison.preference_prob_a > comparison.preference_prob_b:
        print("  Winner: Response B (more detailed)")
    else:
        print("  Winner: Response A (concise)")


# ============================================================================
# Example 4: Explain a Response
# ============================================================================

def example_explain():
    """Get detailed explanation for a response"""
    
    explainer = Explainer(model_path="path/to/model.pt")
    
    context = "User: What movies do you recommend?"
    response = "I recommend Inception - it's a masterpiece with stunning visuals and complex plot."
    
    explanation = explainer.explain(context, response)
    
    print("Explanation Result:")
    print("\nReasoning Trace:")
    for trace in explanation.reasoning_trace:
        print(f"  - {trace}")
    
    print("\nConfidence Signals:")
    for signal, value in explanation.confidence_signals.items():
        print(f"  {signal}: {value:.3f}")
    
    if explanation.weak_signals:
        print("\nWeak Signals (needs attention):")
        for signal in explanation.weak_signals:
            print(f"  - {signal}")


# ============================================================================
# Example 5: Using the Plugin System
# ============================================================================

def example_plugins():
    """Register and use custom plugins"""
    
    from harpo import EvaluatorPlugin, register_evaluator, get_plugin_manager
    
    class CustomSentimentEvaluator(EvaluatorPlugin):
        @property
        def name(self):
            return "sentiment_evaluator"
        
        def score(self, context: str, response: str) -> dict:
            # Simplified example - in reality would use a sentiment model
            positive_words = ["good", "great", "excellent", "amazing"]
            score = sum(1 for word in positive_words if word in response.lower()) / len(positive_words)
            return {"sentiment": min(1.0, score)}
    
    # Register the plugin
    evaluator = CustomSentimentEvaluator()
    register_evaluator(evaluator)
    
    # Use it from the plugin manager
    manager = get_plugin_manager()
    plugin = manager.get_evaluator("sentiment_evaluator")
    
    result = plugin.score("Tell me about movies", "This is great!")
    print(f"Custom Evaluator Result: {result}")


# ============================================================================
# Example 6: REST API Client
# ============================================================================

def example_rest_api():
    """Use HARPO via REST API"""
    
    import requests
    
    # Make sure the API server is running: python -m uvicorn src.api_server:app --port 8000
    
    payload = {
        "context": "What movies do you recommend?",
        "response": "I recommend Inception - it's excellent."
    }
    
    try:
        response = requests.post("http://localhost:8000/evaluate", json=payload)
        result = response.json()
        print("REST API Result:")
        print(json.dumps(result, indent=2))
    except requests.exceptions.ConnectionError:
        print("API server not running. Start it with:")
        print("  python -m uvicorn src.api_server:app --port 8000")


if __name__ == "__main__":
    print("=" * 60)
    print("HARPO API EXAMPLES")
    print("=" * 60)
    
    # Uncomment examples to run them
    # (Requires model path to be set)
    
    # example_score()
    # example_batch_score()
    # example_compare()
    # example_explain()
    # example_plugins()
    # example_rest_api()
    
    print("\nNote: Set model_path to an actual HARPO model checkpoint to run examples.")
    print("See README.md for more details.")
