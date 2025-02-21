from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialisatie van FastAPI
app = FastAPI(title="Chatbot Backend", description="API voor chatbot die vragen beantwoordt op basis van handleidingen.")

# Laad de opgesplitste handleiding
with open("opgesplitste_handleiding.txt", "r", encoding="utf-8") as file:
    handleiding_content = file.readlines()

# Verwerk de handleiding in secties
sections = []
current_section = ""
for line in handleiding_content:
    if line.startswith("### "):
        if current_section:
            sections.append(current_section.strip())
        current_section = line.replace("### ", "").strip()
    else:
        current_section += "\n" + line.strip()
if current_section:
    sections.append(current_section.strip())

# Vectorizer voor tekstvergelijking
vectorizer = TfidfVectorizer()
corpus = [sec for sec in sections]
tfidf_matrix = vectorizer.fit_transform(corpus)

# Vraagmodel
class Question(BaseModel):
    question: str

# Functie om de beste sectie te vinden
def find_best_section(query):
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    best_idx = similarities.argmax()
    return sections[best_idx]

# Functie om antwoord gestructureerd te genereren
def format_answer(query):
    best_section = find_best_section(query)
    steps = best_section.split("\n")
    
    response = f"**{steps[0]}**\n\n"
    response += f"Om {query.lower()} te doen, volg je deze stappen:\n\n"
    
    for i, step in enumerate(steps[1:], start=1):
        response += f"{i}. **{step.strip()}**\n"
    
    response += "\n💡 **Laat me weten als je verdere hulp nodig hebt!**"
    return response

# API Endpoint voor vragen
@app.post("/ask")
def ask_question(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Vraag mag niet leeg zijn.")
    
    answer = format_answer(q.question)
    return {"question": q.question, "answer": answer}

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Automatische documentatie is beschikbaar via /docs
