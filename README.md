# SprintSight API

Predicts Agile sprint **Productivity** and **Quality** using BERTOverflow embeddings + Random Forest.
Based on Sprint2Vec (IEEE TSE 2025).

---

## Project Structure

```
sprintsight_api/
├── artefacts/                  ← Put your .pkl files here
│   ├── scaler.pkl
│   ├── model_productivity.pkl
│   ├── model_quality.pkl
│   └── feature_cols.pkl
├── app/
│   ├── __init__.py
│   ├── main.py                 ← FastAPI app + endpoints
│   ├── schemas.py              ← Request / response models
│   ├── predictor.py            ← Full inference pipeline
│   └── preprocessor.py        ← Text cleaning (matches training)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Place your artefacts
Copy the 4 `.pkl` files downloaded from Colab into the `artefacts/` folder:
```
artefacts/scaler.pkl
artefacts/model_productivity.pkl
artefacts/model_quality.pkl
artefacts/feature_cols.pkl
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will:
- Download and cache BERTOverflow (~440MB on first run only)
- Load all 4 `.pkl` files
- Start accepting requests at `http://localhost:8000`

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Root check |
| GET | `/health` | Check all models are loaded |
| POST | `/predict` | Submit sprint → get predictions |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | ReDoc API documentation |

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "plan_duration_hours": 336,
    "no_issues": 12,
    "no_team_members": 5,
    "no_components": 1.5,
    "fog_index": 12.3,
    "no_comments": 3.0,
    "no_description_changes": 1.0,
    "no_priority_changes": 0.5,
    "no_fix_version_changes": 0.2,
    "dominant_issue_type": "Bug",
    "dominant_priority": "Major",
    "no_distinct_actions": 7.0,
    "developer_activeness": 15.0,
    "dev_preferred_type": "Bug",
    "sprint_text": "Fix login timeout bug affecting mobile users. Improve dashboard load performance. Add CSV export feature."
  }'
```

## Example Response

```json
{
  "productivity": 0.8712,
  "quality": 0.1834,
  "productivity_label": "Good — most committed issues are expected to be completed",
  "quality_label": "Good — low reopen rate expected",
  "embedding_model": "jeniya/BERTOverflow"
}
```

---

## Testing via Swagger UI

1. Start the server
2. Open `http://localhost:8000/docs` in your browser
3. Click **POST /predict → Try it out**
4. Fill in the form and click **Execute**

No code needed — the Swagger UI is a full interactive client.

---

## Connecting a Frontend

The API has CORS enabled, so any frontend on localhost can call it directly:

```javascript
// JavaScript / React example
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ ...sprintData })
});
const result = await response.json();
console.log(result.productivity, result.quality);
```
