#DEBATE BASED DECISION ENGINE (DECISIONAI)

import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
 
# Auto-download required NLTK data (prevents crash on fresh machines)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
 
 
# -------- MODULE 1: INPUT --------
 
def analyze_problem(problem):
    """Extract keywords and detect domain from the problem statement."""
    words = word_tokenize(problem.lower())
    stop_words = set(stopwords.words('english'))
    keywords = [w for w in words if w.isalpha() and w not in stop_words]
 
    # Domain detection — expand keywords based on context
    if any(w in keywords for w in ["eat", "food", "hungry", "meal", "diet", "cook", "nutrition"]):
        keywords += ["food", "hunger", "health"]
 
    if any(w in keywords for w in ["job", "career", "work", "salary", "employment", "office"]):
        keywords += ["career", "money", "future", "stability"]
 
    if any(w in keywords for w in ["study", "college", "degree", "course", "university", "education", "learn"]):
        keywords += ["education", "future", "knowledge"]
 
    if any(w in keywords for w in ["buy", "purchase", "invest", "spend", "money", "cost", "price"]):
        keywords += ["finance", "money", "cost"]
 
    if any(w in keywords for w in ["travel", "trip", "visit", "go", "tour", "vacation", "holiday"]):
        keywords += ["travel", "experience"]
 
    if any(w in keywords for w in ["health", "exercise", "fit", "gym", "medical", "doctor", "treatment"]):
        keywords += ["health", "wellbeing"]
 
    if any(w in keywords for w in ["buy", "laptop", "phone", "gadget", "device", "tech", "software", "app"]):
        keywords += ["technology", "performance"]
 
    if any(w in keywords for w in ["live", "stay", "move", "house", "rent", "city", "location"]):
        keywords += ["lifestyle", "location"]
 
    return list(set(keywords))  # deduplicate
 
 
 
# ══════════════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE BASE
# Maps known field/concept names → their real-world features.
# This is what makes the engine understand abstract options like
# "Robotics", "AI", "Python", "MBA", "Cricket", "Democracy", etc.
# ══════════════════════════════════════════════════════════════════════
 
DOMAIN_KNOWLEDGE = {
 
    # ── TECH FIELDS ──────────────────────────────────────────────────
    "ai":              ["modern", "high demand", "high reward", "scalable", "challenging", "long term benefit"],
    "artificial intelligence": ["modern", "high demand", "high reward", "scalable", "challenging", "long term benefit"],
    "machine learning":["modern", "high demand", "high reward", "scalable", "challenging", "long term benefit"],
    "ml":              ["modern", "high demand", "scalable", "challenging"],
    "deep learning":   ["modern", "high demand", "high reward", "challenging", "long term benefit"],
    "robotics":        ["technical", "hardware", "high demand", "innovative", "challenging", "long term benefit", "high cost"],
    "data science":    ["modern", "high demand", "high reward", "analytical", "long term benefit"],
    "data analytics":  ["modern", "high demand", "analytical", "high reward"],
    "cybersecurity":   ["modern", "high demand", "stable", "high reward", "challenging"],
    "blockchain":      ["modern", "high reward", "high risk", "innovative", "scalable"],
    "cloud computing": ["modern", "scalable", "high demand", "stable", "long term benefit"],
    "iot":             ["modern", "innovative", "scalable", "high demand"],
    "internet of things": ["modern", "innovative", "scalable", "high demand"],
    "ar":              ["modern", "innovative", "high reward", "high risk"],
    "vr":              ["modern", "innovative", "high reward", "high risk"],
    "augmented reality":["modern", "innovative", "high reward"],
    "virtual reality": ["modern", "innovative", "high reward"],
    "web development": ["high demand", "creative", "scalable", "convenient", "long term benefit"],
    "app development": ["high demand", "creative", "scalable", "high reward"],
    "game development":["creative", "high demand", "challenging", "high reward"],
    "devops":          ["modern", "high demand", "stable", "scalable"],
    "embedded systems":["technical", "hardware", "stable", "challenging"],
 
    # ── PROGRAMMING LANGUAGES ────────────────────────────────────────
    "python":          ["easy to learn", "versatile", "high demand", "modern", "long term benefit"],
    "java":            ["stable", "versatile", "high demand", "challenging", "long term benefit"],
    "javascript":      ["high demand", "versatile", "modern", "creative"],
    "c++":             ["high performance", "challenging", "technical", "stable"],
    "c":               ["high performance", "challenging", "technical", "stable"],
    "rust":            ["modern", "high performance", "challenging", "innovative"],
    "golang":          ["modern", "high demand", "scalable"],
    "go":              ["modern", "high demand", "scalable"],
    "kotlin":          ["modern", "convenient", "high demand"],
    "swift":           ["modern", "creative", "high demand"],
    "r":               ["analytical", "educational", "high demand"],
    "matlab":          ["technical", "analytical", "educational", "high cost"],
    "sql":             ["stable", "high demand", "convenient", "educational"],
    "php":             ["convenient", "low cost", "versatile"],
    "ruby":            ["creative", "convenient", "easy to learn"],
 
    # ── EDUCATION DEGREES & STREAMS ──────────────────────────────────
    "mba":             ["high_reward", "social", "long_term_benefit", "high_cost", "networking"],
    "btech":           ["technical", "high_demand", "long_term_benefit", "challenging"],
    "mtech":           ["technical", "high_demand", "long_term_benefit", "educational"],
    "bsc":             ["educational", "affordable", "long_term_benefit"],
    "bca":             ["technical", "affordable", "high_demand"],
    "mca":             ["technical", "high_demand", "long_term_benefit"],
    "phd":             ["educational", "research", "long_term_benefit", "time_consuming", "challenging"],
    "engineering":     ["technical", "high_demand", "challenging", "long_term_benefit"],
    "medical":         ["stable", "high_reward", "long_term_benefit", "challenging", "time_consuming"],
    "law":             ["stable", "high_reward", "challenging", "time_consuming", "long_term_benefit"],
    "commerce":        ["stable", "versatile", "affordable"],
    "arts":            ["creative", "flexible", "affordable"],
    "science":         ["analytical", "educational", "long_term_benefit", "challenging"],
    "humanities":      ["creative", "flexible", "affordable", "social"],
    "design":          ["creative", "high_demand", "innovative"],
    "architecture":    ["creative", "technical", "challenging", "long_term_benefit"],
    "psychology":      ["social", "educational", "long_term_benefit"],
    "economics":       ["analytical", "versatile", "long_term_benefit"],
    "finance":         ["stable", "high_reward", "analytical", "long_term_benefit"],
    "management":      ["high_reward", "social", "long_term_benefit"],
    "marketing":       ["creative", "social", "high_demand"],
    "pharmacy":        ["stable", "long_term_benefit", "challenging"],
    "nursing":         ["stable", "long_term_benefit", "social"],
 
    # ── CAREER / JOB TYPES ───────────────────────────────────────────
    "startup":         ["high_reward", "high_risk", "innovative", "flexible", "challenging"],
    "government job":  ["stable", "low_risk", "long_term_benefit", "secure", "low_reward"],
    "sarkari job":     ["stable", "low_risk", "long_term_benefit", "secure"],
    "corporate job":   ["stable", "high_reward", "challenging", "social"],
    "freelancing":     ["flexible", "independent", "high_risk", "high_reward"],
    "freelance":       ["flexible", "independent", "high_risk", "high_reward"],
    "business":        ["high_reward", "high_risk", "independent", "challenging"],
    "entrepreneurship":["high_reward", "high_risk", "innovative", "independent", "challenging"],
    "remote work":     ["flexible", "convenient", "independent"],
    "work from home":  ["flexible", "convenient", "independent"],
    "research":        ["educational", "analytical", "long_term_benefit", "time_consuming"],
 
    # ── FINANCE / INVESTMENT ─────────────────────────────────────────
    "mutual fund":     ["investment", "low_risk", "long_term_benefit", "high_reward"],
    "stocks":          ["investment", "high_reward", "high_risk"],
    "stock market":    ["investment", "high_reward", "high_risk"],
    "crypto":          ["high_reward", "high_risk", "modern", "volatile"],
    "bitcoin":         ["high_reward", "high_risk", "volatile"],
    "fixed deposit":   ["safe", "low_risk", "stable", "low_reward"],
    "fd":              ["safe", "low_risk", "stable"],
    "ppf":             ["safe", "low_risk", "stable", "long_term_benefit"],
    "gold":            ["safe", "low_risk", "stable"],
    "real estate":     ["stable", "high_reward", "high_cost", "long_term_benefit"],
    "savings":         ["safe", "low_risk", "stable"],
 
    # ── SPORTS & GAMES ───────────────────────────────────────────────
    "cricket":         ["popular", "social", "high_reward", "competitive", "team_sport"],
    "football":        ["popular", "social", "high_reward", "competitive", "team_sport"],
    "basketball":      ["popular", "social", "competitive", "team_sport"],
    "tennis":          ["competitive", "independent", "high_reward", "challenging"],
    "chess":           ["analytical", "independent", "educational", "challenging"],
    "badminton":       ["competitive", "healthy", "convenient"],
    "swimming":        ["healthy", "competitive", "challenging"],
    "running":         ["healthy", "convenient", "low_cost"],
 
    # ── LIFESTYLE / PLACES ───────────────────────────────────────────
    "city":            ["convenient", "social", "high_cost", "modern"],
    "village":         ["peaceful", "low_cost", "healthy", "natural"],
    "metro":           ["convenient", "social", "high_cost", "modern"],
    "abroad":          ["high_cost", "high_reward", "time_consuming", "social", "innovative"],
    "india":           ["low_cost", "stable", "social", "convenient"],
    "online":          ["convenient", "modern", "scalable", "time_saving"],
    "offline":         ["reliable", "social", "traditional"],
 
    # ── FOOD (extended) ──────────────────────────────────────────────
    "home food":       ["healthy", "low_cost", "filling", "nutritious"],
    "home cooked":     ["healthy", "low_cost", "filling", "nutritious"],
    "restaurant":      ["convenient", "social", "high_cost"],
    "zomato":          ["convenient", "time_saving", "high_cost"],
    "swiggy":          ["convenient", "time_saving", "high_cost"],
    "salad":           ["healthy", "light", "nutritious"],
    "pizza":           ["convenient", "unhealthy", "social"],
    "burger":          ["convenient", "unhealthy", "time_saving"],
    "snacks":          ["tasty", "unhealthy", "time_saving","eye catchy"],
    "dal rice":        ["healthy","nutritious","clean","hygienic"],

    # ── OPERATING SYSTEMS / TOOLS ────────────────────────────────────
    "windows":         ["convenient", "versatile", "high_demand", "stable"],
    "linux":           ["free", "low_cost", "technical", "stable", "challenging"],
    "mac":             ["modern", "high_cost", "reliable", "creative"],
    "android":         ["versatile", "low_cost", "high_demand"],
    "ios":             ["modern", "high_cost", "reliable", "secure"],
    "vs code":         ["convenient", "modern", "free", "versatile"],
    "pycharm":         ["convenient", "technical", "educational"],
    "jupyter":         ["convenient", "educational", "analytical"],
 
    # ── SOCIAL / PHILOSOPHICAL ───────────────────────────────────────
    "democracy":       ["social", "stable", "long_term_benefit"],
    "capitalism":      ["high_reward", "high_risk", "innovative"],
    "socialism":       ["stable", "social", "low_risk"],
 
}
 
 
def analyze_option(option):
    """
    Two-layer feature extraction:
    Layer 1 — DOMAIN KNOWLEDGE: checks if the option matches a known field/concept.
    Layer 2 — DESCRIPTIVE KEYWORDS: checks for adjectives like cheap, fast, healthy, etc.
    This makes the engine work for both abstract nouns (Robotics, AI, MBA)
    and descriptive phrases (cheap meal, risky investment, quick snack).
    """
    option_lower = option.lower().strip()
    features = []
 
    # ── LAYER 1: Domain Knowledge Lookup ─────────────────────────────
    # Use whole-word matching so "r" doesn't fire inside "rice",
    # "c" doesn't fire inside "rice", "go" doesn't fire inside "government", etc.
    for domain_key, domain_features in DOMAIN_KNOWLEDGE.items():
        pattern = r'\b' + re.escape(domain_key) + r'\b'
        if re.search(pattern, option_lower):
            features.extend(domain_features)
 
    # ── LAYER 2: Descriptive Keyword Matching ─────────────────────────
 
    # Cost
    if any(w in option_lower for w in ["cheap", "affordable", "free", "budget", "inexpensive", "low cost", "economical"]):
        features.append("low_cost")
    if any(w in option_lower for w in ["expensive", "costly", "premium", "pricey", "high end", "luxury"]):
        features.append("high_cost")
 
    # Time
    if any(w in option_lower for w in ["quick", "fast", "instant", "immediate", "rapid", "short"]):
        features.append("time_saving")
    if any(w in option_lower for w in ["slow", "long", "lengthy", "time consuming", "gradual"]):
        features.append("time_consuming")
 
    # Health
    if any(w in option_lower for w in ["healthy", "nutritious", "organic", "natural", "fresh", "balanced", "wholesome"]):
        features.append("healthy")
    if any(w in option_lower for w in ["unhealthy", "junk", "processed", "fried", "oily", "sugary"]):
        features.append("unhealthy")
 
    # Food
    if any(w in option_lower for w in ["meal", "rice", "roti", "dal", "sabzi", "lunch", "dinner", "breakfast", "thali", "khana"]):
        features.append("filling")
        if "healthy" not in features:
            features.append("nutritious")
    if any(w in option_lower for w in ["snack", "biscuit", "namkeen"]):
        features.append("light")
 
    # Career
    if any(w in option_lower for w in ["job", "career", "employment", "office", "corporate", "profession"]):
        features.append("career_growth")
    if any(w in option_lower for w in ["government", "sarkari", "permanent", "psu", "bank job"]):
        features.append("stable")
        features.append("low_risk")
 
    # Education
    if any(w in option_lower for w in ["degree", "university", "college", "course", "certification", "coaching", "udemy", "coursera", "mooc"]):
        features.append("educational")
        features.append("long_term_benefit")
 
    # Finance
    if any(w in option_lower for w in ["invest", "stock", "equity", "shares", "returns", "portfolio"]):
        features.append("investment")
        features.append("high_reward")
    if any(w in option_lower for w in ["volatile", "speculation", "uncertain"]):
        features.append("high_risk")
 
    # Technology
    if any(w in option_lower for w in ["digital", "app", "software", "cloud", "automation", "saas"]):
        features.append("modern")
        features.append("scalable")
    if any(w in option_lower for w in ["offline", "manual", "traditional", "physical", "paper"]):
        features.append("reliable")
 
    # Risk
    if any(w in option_lower for w in ["safe", "secure", "guaranteed", "insured", "certain"]):
        features.append("low_risk")
        features.append("safe")
    if any(w in option_lower for w in ["risky", "uncertain", "unpredictable", "experimental"]):
        features.append("high_risk")
 
    # Convenience
    if any(w in option_lower for w in ["convenient", "simple", "easy", "accessible", "user friendly", "nearby", "local"]):
        features.append("convenient")
    if any(w in option_lower for w in ["difficult", "complex", "hard", "challenging", "competitive"]):
        features.append("challenging")
 
    # Social
    if any(w in option_lower for w in ["social", "networking", "community", "team", "collaborative"]):
        features.append("social")
    if any(w in option_lower for w in ["solo", "alone", "individual", "independent"]):
        features.append("independent")
 
    return list(set(features))  # deduplicate
 
 
def get_user_input():
    problem = input("Enter your decision problem: ")
    option1 = input("Enter Option 1: ")
    option2 = input("Enter Option 2: ")
    return problem, option1, option2
 
 
# -------- MODULE 2: ARGUMENT GENERATOR --------
 
def generate_arguments(option, keywords, features):
    """
    Generate for/against arguments based on detected features.
    Fully domain-agnostic — works for any topic.
    """
    arguments_for = []
    arguments_against = []
 
    # ── FOR ARGUMENTS ────────────────────────────────
 
    if "low_cost" in features:
        arguments_for.append(f"{option} is budget-friendly and saves money")
    if "time_saving" in features:
        arguments_for.append(f"{option} is quick and saves valuable time")
    if "healthy" in features or "nutritious" in features:
        arguments_for.append(f"{option} is a healthier and more nutritious choice")
    if "filling" in features:
        arguments_for.append(f"{option} is filling and keeps you satisfied longer")
    if "stable" in features:
        arguments_for.append(f"{option} offers long-term stability and security")
    if "low_risk" in features or "safe" in features:
        arguments_for.append(f"{option} is a safe, low-risk option")
    if "high_reward" in features:
        arguments_for.append(f"{option} has high potential for returns and rewards")
    if "career_growth" in features:
        arguments_for.append(f"{option} provides strong career growth opportunities")
    if "educational" in features:
        arguments_for.append(f"{option} builds knowledge and improves qualifications")
    if "long_term_benefit" in features:
        arguments_for.append(f"{option} has significant long-term benefits")
    if "modern" in features:
        arguments_for.append(f"{option} leverages modern technology for better outcomes")
    if "scalable" in features:
        arguments_for.append(f"{option} can scale and grow with future needs")
    if "convenient" in features:
        arguments_for.append(f"{option} is easy to access and convenient to use")
    if "reliable" in features:
        arguments_for.append(f"{option} is dependable and works without external dependencies")
    if "social" in features:
        arguments_for.append(f"{option} provides strong networking and social opportunities")
    if "independent" in features:
        arguments_for.append(f"{option} gives more independence and personal flexibility")
    if "investment" in features:
        arguments_for.append(f"{option} has potential to generate strong financial returns")
    if "high_demand" in features:
        arguments_for.append(f"{option} is in high demand in today's job market")
    if "innovative" in features:
        arguments_for.append(f"{option} is at the cutting edge of innovation")
    if "technical" in features:
        arguments_for.append(f"{option} builds strong technical and practical skills")
    if "versatile" in features:
        arguments_for.append(f"{option} is highly versatile and applicable across many domains")
    if "analytical" in features:
        arguments_for.append(f"{option} sharpens analytical and problem-solving skills")
    if "easy_to_learn" in features:
        arguments_for.append(f"{option} has a gentle learning curve and is beginner-friendly")
    if "creative" in features:
        arguments_for.append(f"{option} encourages creativity and original thinking")
    if "high_performance" in features:
        arguments_for.append(f"{option} delivers superior performance and efficiency")
    if "research" in features:
        arguments_for.append(f"{option} opens strong research and academic pathways")
    if "flexible" in features:
        arguments_for.append(f"{option} offers great flexibility in how and when you work")
    if "popular" in features:
        arguments_for.append(f"{option} is widely popular with a large community and support base")
    if "team_sport" in features or "social" in features:
        arguments_for.append(f"{option} builds teamwork and interpersonal skills")
    if "networking" in features:
        arguments_for.append(f"{option} provides excellent networking and career connections")
 
    # ── AGAINST ARGUMENTS ────────────────────────────
 
    if "high_cost" in features:
        arguments_against.append(f"{option} can be expensive and hard on the budget")
    if "time_consuming" in features:
        arguments_against.append(f"{option} requires a significant time investment")
    if "unhealthy" in features:
        arguments_against.append(f"{option} may have negative health consequences")
    if "light" in features and "food" in keywords:
        arguments_against.append(f"{option} may not be filling enough for your hunger")
    if "high_risk" in features:
        arguments_against.append(f"{option} carries significant risk and uncertainty")
    if "challenging" in features:
        arguments_against.append(f"{option} is difficult and demands a lot of effort to master")
    if "filling" in features and "time" in keywords:
        arguments_against.append(f"{option} might take more time to prepare or consume")
    if "independent" in features:
        arguments_against.append(f"{option} can feel isolating without team or peer support")
    if "technical" in features:
        arguments_against.append(f"{option} has a steep learning curve for beginners")
    if "time_consuming" in features:
        arguments_against.append(f"{option} may take years before showing real results")
    if "high_cost" in features:
        arguments_against.append(f"{option} requires a significant financial investment upfront")
 
    # ── FALLBACKS ────────────────────────────────────
    if not arguments_for:
        arguments_for.append(f"{option} has practical advantages worth considering")
    if not arguments_against:
        arguments_against.append(f"{option} may have some limitations depending on context")
 
    return arguments_for, arguments_against
 
 
# -------- MODULE 3: DEBATE ENGINE --------
 
def run_debate(opt1, opt2, args1_for, args1_against, args2_for, args2_against):
    debate = []
 
    debate.append(f"Round 1 — {opt1} argues FOR itself:")
    for arg in args1_for:
        debate.append(f"   ✅ {arg}")
 
    debate.append(f"Round 1 — {opt2} argues FOR itself:")
    for arg in args2_for:
        debate.append(f"   ✅ {arg}")
 
    debate.append(f"Round 2 — Arguments AGAINST {opt1}:")
    for arg in args1_against:
        debate.append(f"   ❌ {arg}")
 
    debate.append(f"Round 2 — Arguments AGAINST {opt2}:")
    for arg in args2_against:
        debate.append(f"   ❌ {arg}")
 
    # Round 3: Real rebuttals
    debate.append(f"Round 3 — {opt1} rebuts {opt2}:")
    for arg in args2_against:
        debate.append(f"   🔁 Counter: {arg.replace(opt2, opt1)} gives {opt1} an edge here")
 
    debate.append(f"Round 3 — {opt2} rebuts {opt1}:")
    for arg in args1_against:
        debate.append(f"   🔁 Counter: {arg.replace(opt1, opt2)} gives {opt2} an edge here")
 
    debate.append("Round 4 — Final summary: Scoring both options based on all arguments above.")
 
    return debate
 
 
# -------- MODULE 4: SCORING SYSTEM --------
 
def score_arguments(arguments_for, arguments_against, keywords, features):
    score = 0
 
    score += len(arguments_for) * 2
    score -= len(arguments_against) * 1
 
    # Context-aware boosts
    if "healthy" in features or "nutritious" in features:
        score += 2
    if "low_cost" in features:
        score += 1
    if "time_saving" in features:
        score += 1
    if "stable" in features:
        score += 2
    if "low_risk" in features or "safe" in features:
        score += 2
    if "high_reward" in features:
        score += 2
    if "long_term_benefit" in features:
        score += 2
    if "educational" in features:
        score += 1
    if "high_risk" in features:
        score -= 2
    if "unhealthy" in features:
        score -= 2
    if "high_cost" in features:
        score -= 1
    if "time_consuming" in features:
        score -= 1
 
    return score
 
 
# -------- MODULE 5: DECISION ENGINE --------
 
def make_decision(score1, score2, opt1, opt2):
    if score1 == score2:
        return "Both options seem equal", 50
 
    # Normalize to positive range before computing confidence
    min_score = min(score1, score2)
    if min_score < 0:
        score1 -= min_score
        score2 -= min_score
 
    total = score1 + score2
 
    if total == 0:
        return "Both options seem equal", 50
 
    if score1 > score2:
        confidence = int((score1 / total) * 100)
        return opt1, confidence
    else:
        confidence = int((score2 / total) * 100)
        return opt2, confidence
 
 
def generate_explanation(decision, keywords, features1, features2, opt1, opt2):
    if decision not in [opt1, opt2]:
        return "Both options are equally suitable based on current analysis."
 
    chosen = decision
    other = opt2 if decision == opt1 else opt1
    chosen_features = features1 if decision == opt1 else features2
 
    reasons = []
 
    if "healthy" in chosen_features or "nutritious" in chosen_features:
        reasons.append("it is a healthier choice")
    if "low_cost" in chosen_features:
        reasons.append("it is more cost-effective")
    if "time_saving" in chosen_features:
        reasons.append("it saves time")
    if "stable" in chosen_features:
        reasons.append("it offers better stability and security")
    if "low_risk" in chosen_features or "safe" in chosen_features:
        reasons.append("it carries lower risk")
    if "high_reward" in chosen_features:
        reasons.append("it has higher reward potential")
    if "career_growth" in chosen_features:
        reasons.append("it provides better career prospects")
    if "educational" in chosen_features:
        reasons.append("it adds educational and skill value")
    if "long_term_benefit" in chosen_features:
        reasons.append("it has stronger long-term benefits")
    if "convenient" in chosen_features:
        reasons.append("it is more convenient and accessible")
    if "filling" in chosen_features:
        reasons.append("it is more satisfying")
    if "modern" in chosen_features:
        reasons.append("it uses modern and scalable technology")
    if "high_demand" in chosen_features:
        reasons.append("it is highly in demand in the current market")
    if "innovative" in chosen_features:
        reasons.append("it is at the forefront of innovation")
    if "technical" in chosen_features:
        reasons.append("it builds strong technical expertise")
    if "versatile" in chosen_features:
        reasons.append("it is versatile and widely applicable")
    if "analytical" in chosen_features:
        reasons.append("it develops strong analytical thinking")
    if "easy_to_learn" in chosen_features:
        reasons.append("it has a lower learning curve")
    if "creative" in chosen_features:
        reasons.append("it fosters creativity and innovation")
    if "flexible" in chosen_features:
        reasons.append("it offers greater flexibility")
 
    if reasons:
        explanation = f"{chosen} was selected because {', '.join(reasons)}. "
    else:
        explanation = f"{chosen} was selected based on overall argument strength. "
 
    explanation += f"It outperformed {other} in the context of: {', '.join(keywords[:5]) if keywords else 'the given problem'}."
    return explanation
 
 
# -------- MODULE 6: SHOW RESULTS --------
 
def show_results(debate, opt1, opt2, score1, score2, details1, details2, decision, confidence, explanation):
    print("\n========== DEBATE ==========")
    for d in debate:
        print(d)
 
    print("\n========== SCORES ==========")
    print(f"{opt1} : {score1}")
    print(f"{opt2} : {score2}")
 
    print("\n========== FINAL DECISION ==========")
    print("Decision   :", decision)
    print("Confidence :", confidence, "%")
    print("Reason     :", explanation)
 
def generate_arguments_ai(problem, opt1, opt2, api_key):
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = f"""You are a debate engine that generates intelligent arguments for a decision-making system.

Given a problem and two options, generate exactly 3 strong, specific arguments FOR and AGAINST each option.

Problem: {problem}
Option 1: {opt1}
Option 2: {opt2}

Rules:
- Arguments must be specific to the problem and options given
- Each argument should be 1 clear sentence
- Do NOT be generic

Return ONLY a valid JSON object in this exact format, nothing else:
{{
  "opt1_for": ["argument 1", "argument 2", "argument 3"],
  "opt1_against": ["argument 1", "argument 2", "argument 3"],
  "opt2_for": ["argument 1", "argument 2", "argument 3"],
  "opt2_against": ["argument 1", "argument 2", "argument 3"]
}}"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        raw  = response.text.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()

        import json
        data = json.loads(raw)

        return (
            data.get("opt1_for", []),
            data.get("opt1_against", []),
            data.get("opt2_for", []),
            data.get("opt2_against", [])
        )

    except Exception as e:
        print(f"[AI fallback] Error: {e}")
        return None
    
# -------- MAIN FUNCTION --------
def main():
    problem, opt1, opt2 = get_user_input()
    keywords = analyze_problem(problem)
    features1 = analyze_option(opt1)
    features2 = analyze_option(opt2)
    args1_for, args1_against = generate_arguments(opt1, keywords, features1)
    args2_for, args2_against = generate_arguments(opt2, keywords, features2)
    debate = run_debate(opt1, opt2, args1_for, args1_against, args2_for, args2_against)
    score1 = score_arguments(args1_for, args1_against, keywords, features1)
    score2 = score_arguments(args2_for, args2_against, keywords, features2)
    decision, confidence = make_decision(score1, score2, opt1, opt2)
    explanation = generate_explanation(decision, keywords, features1, features2, opt1, opt2)
    show_results(debate, opt1, opt2, score1, score2, {}, {}, decision, confidence, explanation)
 
 
if __name__ == "__main__":
    main()
 