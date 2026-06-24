# SprintSight API

Predicts Agile sprint **Productivity** and **Quality** using BERTOverflow embeddings + Random Forest.
Based on Sprint2Vec (IEEE TSE 2025).

---

## 📂 Project Structure

```text
sprintsight_api/
├── models/
│   ├── scaler.pkl
│   ├── model_productivity.pkl
│   ├── model_quality.pkl
│   └── feature_cols.pkl
├── app/
│   ├── __init__.py
│   ├── main.py                 ← FastAPI application and routing
│   ├── schemas.py              ← Pydantic validation (exact count schema)
│   ├── predictor.py            ← Scaler, Model, and BERT execution logic
│   └── preprocessor.py         ← Text cleaning and Textstat Gunning Fog logic
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Place your models

Ensure your 4 .pkl files are placed inside the models/ directory:

```
models/scaler.pkl
models/model_productivity.pkl
models/model_quality.pkl
models/feature_cols.pkl
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
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

| Method | URL        | Description                     |
| ------ | ---------- | ------------------------------- |
| GET    | `/`        | Root check                      |
| GET    | `/health`  | Check all models are loaded     |
| POST   | `/predict` | Submit sprint → get predictions |
| GET    | `/docs`    | Interactive Swagger UI          |
| GET    | `/redoc`   | ReDoc API documentation         |

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "plan_duration_hours": 336.0,
                "no_issues": 12,
                "no_team_members": 5,
                "no_components": 18.0,
                "no_comments": 36.0,
                "no_description_changes": 4.0,
                "no_priority_changes": 2.0,
                "no_fix_version_changes": 1.0,
                "type_bug_count": 5,
                "type_suggestion_count": 4,
                "type_support_request_count": 3,
                "priority_blocker_count": 0,
                "priority_critical_count": 1,
                "priority_high_count": 2,
                "priority_highest_count": 1,
                "priority_low_count": 2,
                "priority_major_count": 4,
                "priority_medium_count": 2,
                "priority_minor_count": 0,
                "priority_trivial_count": 0,
                "no_distinct_actions": 45.0,
                "developer_activeness": 15.0,
                "dev_prefer_bug_count": 3,
                "dev_prefer_na_count": 1,
                "dev_prefer_subtask_count": 1,
                "dev_prefer_suggestion_count": 0,
                "sprint_text": "Fix login timeout bug affecting mobile users. Improve dashboard load performance. Add export to CSV feature for reports."
  }'
```

## Example Response

```json
{
  "productivity": 0.9077,
  "quality": 0.9069,
  "productivity_label": "Excellent — team is on track to complete nearly all committed work",
  "quality_label": "Very high — majority of completed issues may need rework"
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
