from pathlib import Path

# 1. Define PROJECT_ROOT first so everything below can safely use it!
PROJECT_ROOT = Path(__file__).parent.parent

# 2. Now define all your directories
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

# 3. Create folders if they don't exist yet
for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# 4. Registered models dictionary mapping to Path objects
MODELS = {
    "ridge_regression": {
        "name": "Ridge Baseline",
        "description": "Linear regression baseline model.",
        "path": MODELS_DIR / "ridge_model.pkl",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble bagging regression model.",
        "path": MODELS_DIR / "rf_model.pkl",
    },
    "xgboost_champion": {
        "name": "XGBoost Champion",
        "description": "Gradient boosted trees optimization engine.",
        "path": MODELS_DIR / "xgb_model.json",
    }
}