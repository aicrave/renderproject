from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
import os

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="SEO Entity Extractor API", version="1.0")

# Allow your PHP/JS frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # We can lock this down to "https://webtraffic.blog" later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(payload: TextPayload):
    doc = nlp(payload.text)
    
    entities = []
    entity_freq = Counter([ent.text.lower() for ent in doc.ents if ent.text.strip()])
    total_entities = len(doc.ents)
    
    for ent in doc.ents:
        if not ent.text.strip():
            continue
            
        salience_proxy = entity_freq[ent.text.lower()] / total_entities if total_entities > 0 else 0
        
        entities.append({
            "text": ent.text.strip(),
            "label": ent.label_, 
            "salience_score": round(salience_proxy, 4),
            "frequency": entity_freq[ent.text.lower()]
        })
        
    unique_entities = {v['text'].lower(): v for v in entities}.values()
    sorted_entities = sorted(list(unique_entities), key=lambda x: x['salience_score'], reverse=True)
    
    return {
        "status": "success",
        "metrics": {
            "total_words": len(doc),
            "total_entities_found": len(sorted_entities)
        },
        "entities": sorted_entities
    }
