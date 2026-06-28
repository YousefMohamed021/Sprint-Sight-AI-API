import os
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from app.preprocessor import preprocess_sprint_text , calculate_fog_index
from app.schemas import SprintInput, PredictionResponse


# Paths
BASE_DIR      = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

SCALER_PATH   = MODELS_DIR / "scaler.pkl"
PROD_MODEL    = MODELS_DIR / "model_productivity.pkl"
QUAL_MODEL    = MODELS_DIR / "model_quality.pkl"
FEAT_COLS     = MODELS_DIR / "feature_cols.pkl"

# HuggingFace model ID 
EMBEDDING_MODEL_ID = "jeniya/BERTOverflow"
MAX_TOKENS         = 512

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

        for path in [SCALER_PATH, PROD_MODEL, QUAL_MODEL, FEAT_COLS]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Artefact not found: {path}\n"
                    f"Make sure all .pkl files are in: {MODELS_DIR}"
                )

        print(f"  Loading scaler from {SCALER_PATH.name} ...")
        self.scaler = joblib.load(SCALER_PATH)

        print(f"  Loading productivity model from {PROD_MODEL.name} ...")
        self.model_prod = joblib.load(PROD_MODEL)

        print(f"  Loading quality model from {QUAL_MODEL.name} ...")
        self.model_qual = joblib.load(QUAL_MODEL)

        print(f"  Loading feature column list from {FEAT_COLS.name} ...")
        self.feature_cols: list = joblib.load(FEAT_COLS)

        # Load BERTOverflow
        print(f"  Loading tokenizer: {EMBEDDING_MODEL_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)

        print(f"  Loading BERT model: {EMBEDDING_MODEL_ID} ...")
        self.bert = AutoModel.from_pretrained(EMBEDDING_MODEL_ID)
        self.bert.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert = self.bert.to(self.device)
        print(f"  BERT running on: {self.device}")

        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    # BERTOverflow embeddings
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

        last_hidden = output.last_hidden_state
        mask        = encoded["attention_mask"].unsqueeze(-1).float()
        embedding   = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1) 

        return embedding.cpu().numpy().flatten()        # (768,)

    #  Step 3: Build feature row 
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

        # Sprint features
        row[COL_PLAN_DURATION] = sprint.plan_duration_hours
        row[COL_NO_ISSUE]      = sprint.no_issues
        row[COL_NO_TEAM]       = sprint.no_team_members

        # Issue features
        row[COL_NO_COMPONENT]       = sprint.no_components
        row[COL_FOG_INDEX]          = calculate_fog_index(sprint.sprint_text)
        row[COL_NO_COMMENTS]        = sprint.no_comments
        row[COL_NO_CHANGE_DESC]     = sprint.no_description_changes
        row[COL_NO_CHANGE_PRIORITY] = sprint.no_priority_changes
        row[COL_NO_CHANGE_FIX]      = sprint.no_fix_version_changes

        # EXACT ISSUE TYPE COUNTS 
        row["i_type_Bug"]             = sprint.type_bug_count
        row["i_type_Suggestion"]      = sprint.type_suggestion_count
        row["i_type_Support Request"] = sprint.type_support_request_count

        # EXACT ISSUE PRIORITY COUNTS 
        row["i_priority_Blocker"]  = sprint.priority_blocker_count
        row["i_priority_Critical"] = sprint.priority_critical_count
        row["i_priority_High"]     = sprint.priority_high_count
        row["i_priority_Highest"]  = sprint.priority_highest_count
        row["i_priority_Low"]      = sprint.priority_low_count
        row["i_priority_Major"]    = sprint.priority_major_count
        row["i_priority_Medium"]   = sprint.priority_medium_count
        row["i_priority_Minor"]    = sprint.priority_minor_count
        row["i_priority_Trivial"]  = sprint.priority_trivial_count

#  Developer features
        row[COL_NO_DISTINCT_ACTION]   = sprint.no_distinct_actions
        row[COL_DEVELOPER_ACTIVENESS] = sprint.developer_activeness

        #  Map Developer Preference Counts 
        row["d_most_prefer_type_Bug"]        = sprint.dev_prefer_bug_count
        row["d_most_prefer_type_Na"]         = sprint.dev_prefer_na_count
        row["d_most_prefer_type_Sub-task"]   = sprint.dev_prefer_subtask_count
        row["d_most_prefer_type_Suggestion"] = sprint.dev_prefer_suggestion_count

        for i, val in enumerate(embedding):
            row[f"{EMB_PREFIX}_{i}"] = float(val)

        df = pd.DataFrame([row])

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        df = df[self.feature_cols]

        return df

    #  Main predict method
    def predict(self, sprint: SprintInput) -> PredictionResponse:
        """Full pipeline: SprintInput → PredictionResponse."""

        cleaned_text = preprocess_sprint_text(sprint.sprint_text)

        embedding = self._embed(cleaned_text)

        X_df = self._build_feature_row(sprint, embedding)

        X_scaled = self.scaler.transform(X_df)

        prod_raw = self.model_prod.predict(X_scaled)[0]
        qual_raw = self.model_qual.predict(X_scaled)[0]

        productivity = float(np.clip(np.expm1(prod_raw), 0, None))
        quality      = float(np.clip(np.expm1(qual_raw), 0, None))

        return PredictionResponse(
            productivity=round(productivity, 4)*100,
            quality=round(quality, 4)*100,
        )