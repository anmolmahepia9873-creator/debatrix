from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from majorproject_1 import (
    analyze_problem,
    analyze_option,
    generate_arguments,
    generate_arguments_ai,
    run_debate,
    score_arguments,
    make_decision,
    generate_explanation
)

GEMINI_API_KEY = "AIzaSyCqTKZbdHMdLgLbiEVE6xdbasCaJXUJ2i4"

app = FastAPI(title="Debate Decision Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


class DecisionRequest(BaseModel):
    problem: str
    option1: str
    option2: str


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.post("/analyze")
def analyze(req: DecisionRequest):
    p  = req.problem.strip()
    o1 = req.option1.strip()
    o2 = req.option2.strip()

    keywords  = analyze_problem(p)
    features1 = analyze_option(o1)
    features2 = analyze_option(o2)

    # Try AI first, fallback to rule-based
    ai_result = generate_arguments_ai(p, o1, o2, GEMINI_API_KEY)

    if ai_result:
        args1_for, args1_against, args2_for, args2_against = ai_result
    else:
        args1_for,  args1_against = generate_arguments(o1, keywords, features1)
        args2_for,  args2_against = generate_arguments(o2, keywords, features2)

    debate              = run_debate(o1, o2, args1_for, args1_against, args2_for, args2_against)
    score1              = score_arguments(args1_for, args1_against, keywords, features1)
    score2              = score_arguments(args2_for, args2_against, keywords, features2)
    decision, confidence = make_decision(score1, score2, o1, o2)
    explanation         = generate_explanation(decision, keywords, features1, features2, o1, o2)

    if decision == o1:
        pct1, pct2 = confidence, 100 - confidence
    elif decision == o2:
        pct2, pct1 = confidence, 100 - confidence
    else:
        pct1 = pct2 = 50

    return {
        "decision":    decision,
        "confidence":  confidence,
        "explanation": explanation,
        "option1": {
            "name":         o1,
            "score":        score1,
            "percent":      pct1,
            "features":     features1,
            "args_for":     args1_for,
            "args_against": args1_against,
        },
        "option2": {
            "name":         o2,
            "score":        score2,
            "percent":      pct2,
            "features":     features2,
            "args_for":     args2_for,
            "args_against": args2_against,
        },
        "debate":   debate,
        "keywords": keywords,
    }