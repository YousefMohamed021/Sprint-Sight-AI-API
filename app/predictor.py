import os
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from app.preprocessor import preprocess_sprint_text , calculate_fog_index
from app.schemas import SprintInput, PredictionResponse


# ── Paths
BASE_DIR      = Path(__file__).resolve().parent.parent   # project root
MODELS_DIR = BASE_DIR / "models"

SCALER_PATH   = MODELS_DIR / "scaler.pkl"
PROD_MODEL    = MODELS_DIR / "model_productivity.pkl"
QUAL_MODEL    = MODELS_DIR / "model_quality.pkl"
FEAT_COLS     = MODELS_DIR / "feature_cols.pkl"

# HuggingFace model ID 
EMBEDDING_MODEL_ID = "jeniya/BERTOverflow"
MAX_TOKENS         = 512


# ── Column name constants
# Sprint-level
COL_PLAN_DURATION = "s_plan_duration"
COL_NO_ISSUE      = "s_no_issue"
COL_NO_TEAM       = "s_no_teammember"

# Issue-level
COL_NO_COMPONENT       = "i_no_component"
COL_FOG_INDEX          = "i_fog_index"
COL_NO_COMMENTS        = "i_no_comments"
COL_NO_CHANGE_DESC     = "i_no_change_description"
COL_NO_CHANGE_PRIORITY = "i_no_change_priority"
COL_NO_CHANGE_FIX      = "i_no_change_fix"

# Developer numeric
COL_NO_DISTINCT_ACTION  = "d_no_distinct_action"
COL_DEVELOPER_ACTIVENESS = "d_developer_activeness"

# Embedding prefix
EMB_PREFIX = "bof_emb"

class Predictor:
    """Loads all artefacts at init and exposes a single .predict() method."""

    def __init__(self):
        self._ready = False
        self.embedding_model_id = EMBEDDING_MODEL_ID

        # ── Validate artefact files exist ──────────────────────────────────────
        for path in [SCALER_PATH, PROD_MODEL, QUAL_MODEL, FEAT_COLS]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Artefact not found: {path}\n"
                    f"Make sure all .pkl files are in: {MODELS_DIR}"
                )

        # ── Load sklearn artefacts ─────────────────────────────────────────────
        print(f"  Loading scaler from {SCALER_PATH.name} ...")
        self.scaler = joblib.load(SCALER_PATH)

        print(f"  Loading productivity model from {PROD_MODEL.name} ...")
        self.model_prod = joblib.load(PROD_MODEL)

        print(f"  Loading quality model from {QUAL_MODEL.name} ...")
        self.model_qual = joblib.load(QUAL_MODEL)

        print(f"  Loading feature column list from {FEAT_COLS.name} ...")
        self.feature_cols: list = joblib.load(FEAT_COLS)

        # ── Load BERTOverflow ──────────────────────────────────────────────────
        print(f"  Loading tokenizer: {EMBEDDING_MODEL_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)

        print(f"  Loading BERT model: {EMBEDDING_MODEL_ID} ...")
        self.bert = AutoModel.from_pretrained(EMBEDDING_MODEL_ID)
        self.bert.eval()    # disable dropout for deterministic inference

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert = self.bert.to(self.device)
        print(f"  BERT running on: {self.device}")

        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    # ── Step 2: Generate BERTOverflow embeddings
    def _embed(self, text: str) -> np.ndarray:
        """
        Tokenise cleaned text and return mean-pooled last hidden state.
        Output shape: (768,)  — matches training pipeline exactly.
        """
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.bert(**encoded)

        # Mean-pool over non-padding tokens (attention_mask = 1 for real tokens)
        last_hidden = output.last_hidden_state          # (1, seq_len, 768)
        mask        = encoded["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
        embedding   = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, 768)

        return embedding.cpu().numpy().flatten()        # (768,)

    # ── Step 3: Build feature row ─────────────────────────────────────────────
    def _build_feature_row(
        self,
        sprint: SprintInput,
        embedding: np.ndarray,
    ) -> pd.DataFrame:
        """
        Assembles a single-row DataFrame with every feature in the exact
        column order the scaler and Random Forest were fitted on.
        All missing columns default to 0.
        """
        row = {}

        # ── Sprint features ────────────────────────────────────────────────────
        row[COL_PLAN_DURATION] = sprint.plan_duration_hours
        row[COL_NO_ISSUE]      = sprint.no_issues
        row[COL_NO_TEAM]       = sprint.no_team_members

        # ── Issue features ─────────────────────────────────────────────────────
        row[COL_NO_COMPONENT]       = sprint.no_components
        row[COL_FOG_INDEX]          = calculate_fog_index(sprint.sprint_text)
        row[COL_NO_COMMENTS]        = sprint.no_comments
        row[COL_NO_CHANGE_DESC]     = sprint.no_description_changes
        row[COL_NO_CHANGE_PRIORITY] = sprint.no_priority_changes
        row[COL_NO_CHANGE_FIX]      = sprint.no_fix_version_changes

        # ── EXACT ISSUE TYPE COUNTS ──
        row["i_type_Bug"]             = sprint.type_bug_count
        row["i_type_Suggestion"]      = sprint.type_suggestion_count
        row["i_type_Support Request"] = sprint.type_support_request_count

        # ── EXACT ISSUE PRIORITY COUNTS ──
        row["i_priority_Blocker"]  = sprint.priority_blocker_count
        row["i_priority_Critical"] = sprint.priority_critical_count
        row["i_priority_High"]     = sprint.priority_high_count
        row["i_priority_Highest"]  = sprint.priority_highest_count
        row["i_priority_Low"]      = sprint.priority_low_count
        row["i_priority_Major"]    = sprint.priority_major_count
        row["i_priority_Medium"]   = sprint.priority_medium_count
        row["i_priority_Minor"]    = sprint.priority_minor_count
        row["i_priority_Trivial"]  = sprint.priority_trivial_count

# ── Developer features ─────────────────────────────────────────────────
        row[COL_NO_DISTINCT_ACTION]   = sprint.no_distinct_actions
        row[COL_DEVELOPER_ACTIVENESS] = sprint.developer_activeness

        # ── Map Developer Preference Counts ──
        row["d_most_prefer_type_Bug"]        = sprint.dev_prefer_bug_count
        row["d_most_prefer_type_Na"]         = sprint.dev_prefer_na_count
        row["d_most_prefer_type_Sub-task"]   = sprint.dev_prefer_subtask_count
        row["d_most_prefer_type_Suggestion"] = sprint.dev_prefer_suggestion_count

        # ── Embedding dimensions (768 values) ──────────────────────────────────
        for i, val in enumerate(embedding):
            row[f"{EMB_PREFIX}_{i}"] = float(val)

        # Build DataFrame with correct column order from training
        df = pd.DataFrame([row])

        # Add any missing columns as 0 (safety net)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Enforce exact column order
        df = df[self.feature_cols]

        return df

    # ── Human-readable labels ─────────────────────────────────────────────────
    @staticmethod
    def _productivity_label(score: float) -> str:
        if score >= 0.90:  return "Excellent — team is on track to complete nearly all committed work"
        if score >= 0.75:  return "Good — most committed issues are expected to be completed"
        if score >= 0.50:  return "Moderate — roughly half the committed issues will be completed"
        if score >= 0.25:  return "Low — significant underdelivery expected"
        return                    "Very low — most committed issues are at risk of not being completed"

    @staticmethod
    def _quality_label(score: float) -> str:
        if score <= 0.05:  return "Excellent — very few issues expected to be reopened"
        if score <= 0.15:  return "Good — low reopen rate expected"
        if score <= 0.30:  return "Moderate — some rework likely"
        if score <= 0.50:  return "High — significant rework risk"
        return                    "Very high — majority of completed issues may need rework"

    # ── Main predict method ───────────────────────────────────────────────────
    def predict(self, sprint: SprintInput) -> PredictionResponse:
        """Full pipeline: SprintInput → PredictionResponse."""

        # Step 1: Clean text (matches training preprocessing exactly)
        cleaned_text = preprocess_sprint_text(sprint.sprint_text)

        # Step 2: Generate BERTOverflow embedding
        embedding = self._embed(cleaned_text)                   # (768,)

        # Step 3: Assemble feature row in correct column order
        X_df = self._build_feature_row(sprint, embedding)       # (1, 796)

        # Step 4: Apply the same StandardScaler fitted during training
        X_scaled = self.scaler.transform(X_df)                  # (1, 796)

        # Step 5 & 6: Predict and inverse log1p transform
        prod_raw = self.model_prod.predict(X_scaled)[0]
        qual_raw = self.model_qual.predict(X_scaled)[0]

        productivity = float(np.clip(np.expm1(prod_raw), 0, None))
        quality      = float(np.clip(np.expm1(qual_raw), 0, None))

        return PredictionResponse(
            productivity=round(productivity, 4),
            quality=round(quality, 4),
            productivity_label=self._productivity_label(productivity),
            quality_label=self._quality_label(quality),
        )