import os

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

REASON_LOAD_BIG_MODEL = "Expensive test, requires loading a large trained model, skip in CI."
REASON_LOAD_BIG_DATA = "Expensive test, requires loading datasets, skip in CI."
