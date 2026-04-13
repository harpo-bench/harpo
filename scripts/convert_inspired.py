#!/usr/bin/env python3
"""
HARPO: INSPIRED Dataset Converter

Converts INSPIRED (INteraction-based movie Search through dIalog with Retrieval and rEcommendation) 
dataset to HARPO format.

INSPIRED is more sociable than ReDial - recommenders share personal experiences and engage socially.
Dataset source: https://github.com/sweetpeach/Inspired

Key differences from ReDial:
- More social/emotional language
- Recommenders share personal opinions
- Richer contextual information
- Smaller dataset (~1K conversations)

Usage:
    python convert_inspired.py --output ./inspired_harpo_data --use-llm --api-key YOUR_KEY
"""

import json
import os
import re
import random
import time
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

# All available VTOs
ALL_VTOS = [
    "extract_context", "extract_entities", "retrieve_preferences", 
    "identify_constraints", "search_candidates", "filter_results",
    "rank_options", "compare_options", "explain_choice", "refine_query",
    "analyze_sentiment", "model_user_state"
]

# ============================================================================
# LLM PROMPTS (Adapted for INSPIRED's sociable style)
# ============================================================================

CLASSIFY_UTTERANCE_PROMPT = """Classify this utterance in a sociable movie recommendation conversation.

Context: {context}
Current utterance: {utterance}
Speaker: {speaker}
Contains movie mention: {has_movie}

The INSPIRED dataset features more personal, sociable conversations where recommenders
share opinions and experiences. Classify into ONE of these categories:

- greeting: Initial greeting or social opener
- ask_preference: Recommender asking what user wants/likes
- provide_preference: User stating preferences or interests
- share_experience: Sharing personal movie-watching experience
- recommend: Suggesting specific movies
- explain: Providing information or reasons about movies
- social_chat: General social conversation (not about movies)
- accept: User accepting/liking a recommendation
- reject: User rejecting/disliking a recommendation
- ask_info: Asking for more details
- provide_info: Giving details about a movie
- empathize: Showing understanding or relating to user
- thank: Thanking
- goodbye: Ending conversation

Return ONLY the category name, nothing else."""

VTO_ASSIGNMENT_PROMPT = """Assign Virtual Tool Operations (VTOs) for this recommendation response.

Context: {context}
User input: {user_input}
System response: {response}
Utterance type: {utterance_type}

The INSPIRED dataset features sociable, personal conversations. Consider emotional and social aspects.

Available VTOs:
- analyze_sentiment: Understand user emotions, mood from message
- extract_context: Extract situation/occasion from conversation
- extract_entities: Identify movies, genres, actors mentioned
- retrieve_preferences: Get user's stated preferences
- identify_constraints: Find requirements (genre, mood, occasion)
- model_user_state: Build model of user's current needs/state
- search_candidates: Search for matching movies
- filter_results: Apply filters to narrow down
- rank_options: Order movies by relevance
- compare_options: Compare multiple movies
- explain_choice: Explain why recommending with personal touch
- refine_query: Ask clarifying questions

Return a comma-separated list of 1-5 most relevant VTOs for this response.
Consider both recommendation AND social aspects.
Example: analyze_sentiment, extract_context, search_candidates, explain_choice"""

GENERATE_REJECTED_PROMPT = """Generate a low-quality response for preference learning.

Context: {context}
User: {user_input}
High-quality response: {chosen}
Movie being recommended: {movie}

Generate a response that is WORSE because it:
- Ignores the social/emotional context
- Gives generic recommendations without personal touch
- Misses user preferences
- Lacks explanation or enthusiasm
- Feels impersonal or robotic

Low-quality response:"""


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """OpenAI client for data generation with parallel processing support"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", 
                 max_workers: int = 10, rate_limit_per_min: int = 500):
        self.model = model
        self.api_key = api_key
        self._client = None
        self.max_workers = max_workers
        self.rate_limit_per_min = rate_limit_per_min
        self._request_times = []
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                
                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or use --api-key")
                
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client
    
    def _rate_limit_wait(self):
        """Simple rate limiting"""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.rate_limit_per_min:
            sleep_time = 60 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._request_times.append(time.time())
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 1024, json_mode: bool = False) -> str:
        """Generate text from prompt (single request)"""
        self._rate_limit_wait()
        client = self._get_client()
        
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return ""
    
    def generate_batch(self, prompts: List[str], temperature: float = 0.7,
                       max_tokens: int = 256, json_mode: bool = False,
                       show_progress: bool = True) -> List[str]:
        """Generate responses for multiple prompts in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [""] * len(prompts)
        
        def process_single(idx_prompt):
            idx, prompt = idx_prompt
            try:
                self._rate_limit_wait()
                response = self.generate(prompt, temperature, max_tokens, json_mode)
                return idx, response
            except Exception as e:
                print(f"Error processing prompt {idx}: {e}")
                return idx, ""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single, (i, p)): i 
                      for i, p in enumerate(prompts)}
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(prompts), desc="LLM batch processing")
            
            for future in iterator:
                try:
                    idx, response = future.result(timeout=60)
                    results[idx] = response
                except Exception as e:
                    print(f"Future error: {e}")
        
        return results


# ============================================================================
# DATA LOADING
# ============================================================================

def parse_inspired_tsv(tsv_content: str) -> List[Dict]:
    """
    Parses the raw TSV content.
    Confirmed Mapping: Col 0=ID, Col 2=Speaker, Col 4=Text
    """
    conversations = []
    current_conv = {}
    current_messages = []
    last_conv_id = None
    
    lines = tsv_content.strip().split('\n')
    start_idx = 1 if lines and "dialog_id" in lines[0].lower() else 0

    for line in lines[start_idx:]:
        parts = line.split('\t')
        if len(parts) < 3: continue
            
        conv_id = parts[0].strip()
        role_raw = parts[2].strip()
        
        # TEXT EXTRACTION (Corrected based on logs)
        if len(parts) >= 5:
            text = parts[4].strip()
        elif len(parts) == 4:
            text = parts[3].strip()
        else:
            text = parts[-1].strip()

        # SPEAKER NORMALIZATION
        if role_raw.upper() == "SEEKER": speaker = "user"
        elif role_raw.upper() == "RECOMMENDER": speaker = "recommender"
        else: speaker = role_raw.lower()

        if conv_id != last_conv_id:
            if last_conv_id is not None:
                current_conv["messages"] = current_messages
                conversations.append(current_conv)
            current_conv = {"conversationId": conv_id, "messages": []}
            current_messages = []
            last_conv_id = conv_id

        current_messages.append({"role": speaker, "text": text})

    if current_conv and current_messages:
        current_conv["messages"] = current_messages
        conversations.append(current_conv)

    return conversations

def load_inspired_from_github() -> Tuple[List[Dict], List[Dict]]:
    base_url = "https://raw.githubusercontent.com/sweetpeach/Inspired/master/data/dialog_data"
    files = {"train": f"{base_url}/train.tsv", "dev": f"{base_url}/dev.tsv", "test": f"{base_url}/test.tsv"}
    
    print("Downloading INSPIRED dataset...")
    data_splits = {}
    
    for split, url in files.items():
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')
                data_splits[split] = parse_inspired_tsv(content)
                print(f"  ✓ Loaded {split}: {len(data_splits[split])} conversations")
        except Exception as e:
            print(f"  ✗ Failed {split}: {e}")
            data_splits[split] = []

    return data_splits.get("train", []) + data_splits.get("dev", []), data_splits.get("test", [])


def load_inspired_local(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load INSPIRED from local directory if GitHub fails."""
    
    train_path = os.path.join(data_dir, "train.json")
    dev_path = os.path.join(data_dir, "dev.json")
    test_path = os.path.join(data_dir, "test.json")
    
    train_data = []
    dev_data = []
    test_data = []
    
    if os.path.exists(train_path):
        with open(train_path) as f:
            train_data = json.load(f)
    
    if os.path.exists(dev_path):
        with open(dev_path) as f:
            dev_data = json.load(f)
    
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_data = json.load(f)
    
    return train_data + dev_data, test_data


# ============================================================================
# MOVIE DATABASE
# ============================================================================

def build_movie_database(conversations: List[Dict]) -> Dict[str, str]:
    """Build movie ID to name mapping from INSPIRED data."""
    
    movie_db = {}
    
    for conv in conversations:
        # INSPIRED format has movies in different places
        if "movies" in conv:
            for movie_id, movie_name in conv["movies"].items():
                movie_db[movie_id] = movie_name
        
        # Also extract from messages
        messages = conv.get("messages", conv.get("dialog", []))
        for msg in messages:
            # Check for movie mentions in text (format: @MOVIE_ID)
            text = msg.get("text", msg.get("utterance", ""))
            movie_mentions = re.findall(r'@(\d+)', text)
            
            # Get movie names from conversation metadata
            conv_movies = conv.get("movieMentions", conv.get("movies", {}))
            for movie_id in movie_mentions:
                if movie_id in conv_movies:
                    movie_db[movie_id] = conv_movies[movie_id]
    
    return movie_db


def extract_movie_from_text(text: str, movie_db: Dict[str, str]) -> Optional[str]:
    """Extract movie name from text containing @MOVIE_ID."""
    
    # Find @ID mentions
    mentions = re.findall(r'@(\d+)', text)
    
    for movie_id in mentions:
        if movie_id in movie_db:
            return movie_db[movie_id]
    
    return None


def replace_movie_ids(text: str, movie_db: Dict[str, str]) -> str:
    """Replace @MOVIE_ID with actual movie names."""
    
    def replace_func(match):
        movie_id = match.group(1)
        if movie_id in movie_db:
            return f'"{movie_db[movie_id]}"'
        return match.group(0)
    
    return re.sub(r'@(\d+)', replace_func, text)


# ============================================================================
# UTTERANCE PROCESSING
# ============================================================================

def extract_utterances_from_conversation(conv: Dict, movie_db: Dict[str, str]) -> List[Dict]:
    """Extract turn-by-turn utterances from INSPIRED conversation."""
    
    utterances = []
    conv_id = conv.get("conversationId", conv.get("conv_id", str(random.randint(1000, 9999))))
    
    # Get messages (INSPIRED uses different field names)
    messages = conv.get("messages", conv.get("dialog", []))
    
    if not messages:
        return []
    
    # Build conversation context
    context_turns = []
    
    for i, msg in enumerate(messages):
        # Get speaker (INSPIRED format)
        speaker = msg.get("role", msg.get("sender", "unknown"))
        if speaker in ["SEEKER", "seeker", "user", "0"]:
            speaker = "user"
        elif speaker in ["RECOMMENDER", "recommender", "assistant", "1"]:
            speaker = "recommender"
        
        # Get text
        text = msg.get("text", msg.get("utterance", ""))
        if not text.strip():
            continue
        
        # Replace movie IDs with names
        text_clean = replace_movie_ids(text, movie_db)
        
        # Check for movie mention
        movie_mentioned = extract_movie_from_text(text, movie_db)
        
        # Build context from previous turns
        context = "\n".join(context_turns[-6:]) if context_turns else ""
        
        # For recommender turns, get previous user input
        user_input = ""
        if speaker == "recommender" and context_turns:
            for prev in reversed(context_turns):
                if prev.startswith("User:"):
                    user_input = prev.replace("User: ", "")
                    break
        
        utterances.append({
            "conversation_id": conv_id,
            "turn_idx": i,
            "speaker": speaker,
            "text": text_clean,
            "context": context,
            "user_input": user_input,
            "has_movie": movie_mentioned is not None,
            "movie_mentioned": movie_mentioned,
            "utterance_type": None,  # To be classified
            "vtos": []  # To be assigned
        })
        
        # Add to context
        speaker_label = "User" if speaker == "user" else "Assistant"
        context_turns.append(f"{speaker_label}: {text_clean}")
    
    return utterances


# ============================================================================
# CLASSIFICATION & VTO ASSIGNMENT
# ============================================================================

def classify_utterance_heuristic(utt: Dict) -> str:
    """Classify utterance using heuristics (fallback when no LLM)."""
    
    text = utt["text"].lower()
    speaker = utt["speaker"]
    has_movie = utt["has_movie"]
    
    # Greeting patterns
    if any(g in text for g in ["hi ", "hello", "hey ", "good morning", "good evening"]):
        return "greeting"
    
    # Goodbye patterns
    if any(g in text for g in ["bye", "goodbye", "see you", "take care", "have a good"]):
        return "goodbye"
    
    # Thank patterns
    if any(t in text for t in ["thank", "thanks", "appreciate"]):
        return "thank"
    
    if speaker == "user":
        # User patterns
        if "?" in text:
            return "ask_info"
        if any(p in text for p in ["i like", "i love", "i enjoy", "i want", "looking for", "in the mood"]):
            return "provide_preference"
        if any(a in text for a in ["sounds good", "i'll watch", "great choice", "perfect"]):
            return "accept"
        if any(r in text for r in ["not really", "don't like", "not interested", "something else"]):
            return "reject"
        return "provide_preference"
    
    else:  # recommender
        if has_movie:
            if any(e in text for e in ["because", "it's about", "it has", "you might like", "i think"]):
                return "explain"
            return "recommend"
        if "?" in text:
            return "ask_preference"
        if any(s in text for s in ["i ", "me too", "i also", "personally", "i love", "i watched"]):
            return "share_experience"
        if any(e in text for e in ["i understand", "that makes sense", "i know what you mean"]):
            return "empathize"
        return "social_chat"


def assign_vtos_heuristic(utt: Dict) -> List[str]:
    """Assign VTOs using heuristics based on utterance type."""
    
    utt_type = utt.get("utterance_type", "")
    has_movie = utt.get("has_movie", False)
    
    vto_map = {
        "greeting": ["extract_context"],
        "ask_preference": ["extract_context", "refine_query"],
        "provide_preference": ["extract_entities", "retrieve_preferences"],
        "share_experience": ["analyze_sentiment", "model_user_state"],
        "recommend": ["search_candidates", "rank_options", "explain_choice"],
        "explain": ["extract_entities", "explain_choice"],
        "social_chat": ["analyze_sentiment", "model_user_state"],
        "accept": ["model_user_state"],
        "reject": ["analyze_sentiment", "refine_query"],
        "ask_info": ["extract_entities"],
        "provide_info": ["extract_entities", "explain_choice"],
        "empathize": ["analyze_sentiment", "model_user_state"],
        "thank": ["model_user_state"],
        "goodbye": ["model_user_state"]
    }
    
    vtos = vto_map.get(utt_type, ["extract_context"])
    
    # Add VTOs based on content
    if has_movie:
        if "search_candidates" not in vtos:
            vtos.append("search_candidates")
    
    return vtos[:5]  # Limit to 5 VTOs


def process_utterances_with_llm(utterances: List[Dict], llm: LLMClient) -> List[Dict]:
    """Process utterances with LLM for classification and VTO assignment."""
    
    print("  Classifying utterances with LLM...")
    
    # Build classification prompts
    classify_prompts = []
    for utt in utterances:
        if utt["speaker"] == "recommender":
            prompt = CLASSIFY_UTTERANCE_PROMPT.format(
                context=utt["context"][:500],
                utterance=utt["text"][:300],
                speaker=utt["speaker"],
                has_movie=utt["has_movie"]
            )
            classify_prompts.append(prompt)
        else:
            classify_prompts.append("")
    
    # Batch classify
    non_empty_indices = [i for i, p in enumerate(classify_prompts) if p]
    non_empty_prompts = [classify_prompts[i] for i in non_empty_indices]
    
    if non_empty_prompts:
        responses = llm.generate_batch(non_empty_prompts, temperature=0.3, max_tokens=50)
        
        for idx, resp in zip(non_empty_indices, responses):
            resp_clean = resp.strip().lower().replace("_", " ").replace("-", " ")
            # Map to valid category
            valid_cats = ["greeting", "ask_preference", "provide_preference", "share_experience",
                         "recommend", "explain", "social_chat", "accept", "reject", 
                         "ask_info", "provide_info", "empathize", "thank", "goodbye"]
            
            for cat in valid_cats:
                if cat.replace("_", " ") in resp_clean:
                    utterances[idx]["utterance_type"] = cat
                    break
            else:
                utterances[idx]["utterance_type"] = classify_utterance_heuristic(utterances[idx])
    
    # Classify remaining with heuristics
    for utt in utterances:
        if not utt.get("utterance_type"):
            utt["utterance_type"] = classify_utterance_heuristic(utt)
    
    # Assign VTOs with LLM for recommender turns
    print("  Assigning VTOs with LLM...")
    
    vto_prompts = []
    vto_indices = []
    
    for i, utt in enumerate(utterances):
        if utt["speaker"] == "recommender" and utt.get("user_input"):
            prompt = VTO_ASSIGNMENT_PROMPT.format(
                context=utt["context"][:400],
                user_input=utt["user_input"][:200],
                response=utt["text"][:300],
                utterance_type=utt["utterance_type"]
            )
            vto_prompts.append(prompt)
            vto_indices.append(i)
    
    if vto_prompts:
        vto_responses = llm.generate_batch(vto_prompts, temperature=0.3, max_tokens=100)
        
        for idx, resp in zip(vto_indices, vto_responses):
            # Parse VTO list
            vtos = []
            resp_lower = resp.lower()
            for vto in ALL_VTOS:
                if vto in resp_lower:
                    vtos.append(vto)
            
            if vtos:
                utterances[idx]["vtos"] = vtos[:5]
            else:
                utterances[idx]["vtos"] = assign_vtos_heuristic(utterances[idx])
    
    # Assign VTOs for remaining
    for utt in utterances:
        if not utt.get("vtos"):
            utt["vtos"] = assign_vtos_heuristic(utt)
    
    return utterances


def process_utterances_heuristic(utterances: List[Dict]) -> List[Dict]:
    """Process utterances with heuristics only."""
    
    for utt in utterances:
        utt["utterance_type"] = classify_utterance_heuristic(utt)
        utt["vtos"] = assign_vtos_heuristic(utt)
    
    return utterances


# ============================================================================
# SFT EXAMPLE CREATION
# ============================================================================

def create_sft_examples(utterances: List[Dict]) -> List[Dict]:
    """Create SFT training examples from processed utterances."""
    
    examples = []
    
    for utt in utterances:
        # Only create examples for recommender turns with user input
        if utt["speaker"] != "recommender" or not utt.get("user_input"):
            continue
        
        # Build input
        input_text = "<|domain:movies|>\n\n"
        if utt["context"]:
            input_text += utt["context"] + "\n"
        input_text += f"User: {utt['user_input']}"
        
        # Build output with VTO think tags
        vto_str = ", ".join(utt["vtos"]) if utt["vtos"] else "extract_context"
        output_text = f"<|think|>{vto_str}<|/think|>\n{utt['text']}"
        
        example = {
            "input": input_text,
            "output": output_text,
            "vtos": utt["vtos"],
            "domain": "movies",
            "conversation_id": utt["conversation_id"],
            "utterance_type": utt["utterance_type"]
        }
        
        # Add ground truth if movie was mentioned
        if utt.get("movie_mentioned"):
            example["ground_truth_item"] = utt["movie_mentioned"]
        
        examples.append(example)
    
    return examples


# ============================================================================
# PREFERENCE PAIR GENERATION
# ============================================================================

def generate_preference_pairs(examples: List[Dict], all_movies: List[str],
                             llm: Optional[LLMClient] = None) -> List[Dict]:
    """Generate preference pairs for CHARM training."""
    
    preference_pairs = []
    
    # Filter examples with ground truth
    examples_with_gt = [ex for ex in examples if ex.get("ground_truth_item")]
    
    print(f"  Examples with ground truth: {len(examples_with_gt)}")
    
    if llm:
        print("  Generating rejected responses with LLM...")
        
        # Build prompts
        prompts = []
        for ex in examples_with_gt[:500]:  # Limit for API cost
            prompt = GENERATE_REJECTED_PROMPT.format(
                context=ex["input"][:400],
                user_input=ex["input"].split("User: ")[-1][:200],
                chosen=ex["output"][:300],
                movie=ex.get("ground_truth_item", "unknown")
            )
            prompts.append(prompt)
        
        responses = llm.generate_batch(prompts, temperature=0.8, max_tokens=200)
        
        for ex, rejected_text in zip(examples_with_gt[:500], responses):
            if not rejected_text.strip():
                rejected_text = "I'm not sure what to recommend. There are lots of good movies out there."
            
            preference_pairs.append({
                "conversation_id": ex["conversation_id"],
                "context": ex["input"],
                "chosen": ex["output"],
                "rejected": rejected_text,
                "chosen_vtos": ex["vtos"],
                "rejected_vtos": ["search_candidates"],
                "reward_margin": 0.3 + random.random() * 0.4,
                "ground_truth_movie": ex.get("ground_truth_item"),
                "hierarchical_rewards": {
                    "relevance": {"chosen": 0.9, "rejected": 0.3},
                    "diversity": {"chosen": 0.8, "rejected": 0.4},
                    "user_satisfaction": {"chosen": 0.85, "rejected": 0.35},
                    "engagement": {"chosen": 0.9, "rejected": 0.5}
                }
            })
    
    else:
        print("  Generating rejected responses (heuristic)...")
        
        # Generic bad responses for INSPIRED (less personalized)
        generic_bad = [
            "I'd recommend checking out some popular movies.",
            "There are many good options. Let me know if you want specific suggestions.",
            "That's a good choice. Have you considered other genres?",
            "I'm not sure about that one. Maybe try something else?",
            "There are lots of movies like that. Hard to pick just one.",
            "Movies are great entertainment. What kind are you looking for?",
        ]
        
        for ex in tqdm(examples_with_gt, desc="Creating preference pairs"):
            chosen = ex["output"]
            gt_movie = ex.get("ground_truth_item")
            
            # Strategy selection
            strategy = random.random()
            
            if strategy < 0.4 and gt_movie:
                # Replace with wrong movie
                wrong_movies = [m for m in all_movies if m != gt_movie]
                if wrong_movies:
                    wrong_movie = random.choice(wrong_movies)
                    rejected = chosen.replace(f'"{gt_movie}"', f'"{wrong_movie}"')
                    if rejected == chosen:
                        rejected = random.choice(generic_bad)
                else:
                    rejected = random.choice(generic_bad)
            elif strategy < 0.7:
                # Generic response (no personal touch)
                rejected = random.choice(generic_bad)
            else:
                # Truncated/incomplete
                if len(chosen) > 100:
                    rejected = chosen[:len(chosen)//3] + "..."
                else:
                    rejected = random.choice(generic_bad)
            
            preference_pairs.append({
                "conversation_id": ex["conversation_id"],
                "context": ex["input"],
                "chosen": chosen,
                "rejected": rejected,
                "chosen_vtos": ex["vtos"],
                "rejected_vtos": ["search_candidates"],
                "reward_margin": 0.3 + random.random() * 0.4,
                "ground_truth_movie": gt_movie,
                "hierarchical_rewards": {
                    "relevance": {"chosen": 0.9, "rejected": 0.3},
                    "diversity": {"chosen": 0.8, "rejected": 0.4},
                    "user_satisfaction": {"chosen": 0.85, "rejected": 0.35},
                    "engagement": {"chosen": 0.9, "rejected": 0.5}
                }
            })
    
    return preference_pairs


# ============================================================================
# MAIN CONVERSION
# ============================================================================

def convert_inspired_to_harpo(output_dir: str, max_train: int = None, max_test: int = None,
                              llm: Optional[LLMClient] = None, local_data_dir: str = None):
    """Main conversion function for INSPIRED dataset."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("INSPIRED to HARPO Conversion")
    print("=" * 60)
    print(f"Using LLM: {llm is not None}")
    print(f"Output: {output_dir}")
    print("")
    
    # Load data
    if local_data_dir and os.path.exists(local_data_dir):
        print(f"Loading from local directory: {local_data_dir}")
        train_raw, test_raw = load_inspired_local(local_data_dir)
    else:
        train_raw, test_raw = load_inspired_from_github()
    
    if not train_raw:
        print("ERROR: Could not load INSPIRED data")
        print("Please download manually from: https://github.com/sweetpeach/Inspired")
        return None
    
    if max_train:
        train_raw = train_raw[:max_train]
    if max_test:
        test_raw = test_raw[:max_test]
    
    # Build movie database
    print("\nBuilding movie database...")
    movie_db = build_movie_database(train_raw + test_raw)
    print(f"  Found {len(movie_db)} unique movies")
    
    # Process training data
    print("\nProcessing training conversations...")
    all_train_utterances = []
    for conv in tqdm(train_raw, desc="Extracting utterances"):
        utterances = extract_utterances_from_conversation(conv, movie_db)
        all_train_utterances.extend(utterances)
    
    print(f"  Total utterances: {len(all_train_utterances)}")
    
    # Classify and assign VTOs
    if llm:
        all_train_utterances = process_utterances_with_llm(all_train_utterances, llm)
    else:
        all_train_utterances = process_utterances_heuristic(all_train_utterances)
    
    # Create SFT examples
    print("\nCreating SFT examples...")
    train_sft = create_sft_examples(all_train_utterances)
    print(f"  Training examples: {len(train_sft)}")
    
    # Process test data
    print("\nProcessing test conversations...")
    all_test_utterances = []
    for conv in tqdm(test_raw, desc="Extracting test utterances"):
        utterances = extract_utterances_from_conversation(conv, movie_db)
        all_test_utterances.extend(utterances)
    
    if llm:
        all_test_utterances = process_utterances_with_llm(all_test_utterances, llm)
    else:
        all_test_utterances = process_utterances_heuristic(all_test_utterances)
    
    test_sft = create_sft_examples(all_test_utterances)
    print(f"  Test examples: {len(test_sft)}")
    
    # Get all movie names
    all_movies = list(set(movie_db.values()))
    
    # Generate preference pairs
    print("\nGenerating preference pairs...")
    train_pref = generate_preference_pairs(train_sft, all_movies, llm)
    print(f"  Preference pairs: {len(train_pref)}")
    
    # Save files
    print("\nSaving files...")
    
    with open(os.path.join(output_dir, "sft_data.json"), "w") as f:
        json.dump(train_sft, f, indent=2)
    
    with open(os.path.join(output_dir, "test_sft.json"), "w") as f:
        json.dump(test_sft, f, indent=2)
    
    with open(os.path.join(output_dir, "preference_data.json"), "w") as f:
        json.dump(train_pref, f, indent=2)
    
    with open(os.path.join(output_dir, "movie_list.json"), "w") as f:
        json.dump(all_movies, f, indent=2)
    
    stats = {
        "dataset": "INSPIRED",
        "num_train_sft": len(train_sft),
        "num_test_sft": len(test_sft),
        "num_preference_pairs": len(train_pref),
        "num_movies": len(all_movies),
        "examples_with_recommendations": sum(1 for ex in train_sft if ex.get("ground_truth_item")),
        "used_llm": llm is not None
    }
    
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Training examples: {stats['num_train_sft']}")
    print(f"Test examples: {stats['num_test_sft']}")
    print(f"Preference pairs: {stats['num_preference_pairs']}")
    print(f"Total movies: {stats['num_movies']}")
    print(f"Examples with recommendations: {stats['examples_with_recommendations']}")
    print(f"Used LLM: {stats['used_llm']}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert INSPIRED to HARPO format")
    parser.add_argument("--output", default="./inspired_harpo_data", help="Output directory")
    parser.add_argument("--local-data", default=None, help="Local data directory (if GitHub fails)")
    parser.add_argument("--max-train", type=int, default=None, help="Max training conversations")
    parser.add_argument("--max-test", type=int, default=None, help="Max test conversations")
    parser.add_argument("--use-llm", action="store_true", 
                        help="Use GPT-4o-mini for classification and VTO assignment")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    llm = None
    if args.use_llm:
        print("Initializing LLM client...")
        try:
            llm = LLMClient(api_key=args.api_key, model=args.model)
            llm._get_client()
            print(f"  ✓ Connected to OpenAI ({args.model})")
        except Exception as e:
            print(f"  ✗ Failed to initialize LLM: {e}")
            print("  Falling back to heuristic processing")
            llm = None
    
    convert_inspired_to_harpo(
        args.output, 
        args.max_train, 
        args.max_test, 
        llm,
        args.local_data
    )
