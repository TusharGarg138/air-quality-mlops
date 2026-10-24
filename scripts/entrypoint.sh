#!/bin/bash
set -e

echo "Starting container initialization..."

# Verify model artifact exists; attempt pull from DVC remote if missing
if [ ! -s "models/best_model.pkl" ] || [ ! -s "models/preprocessor.pkl" ] || [ ! -s "models/model_metadata.json" ]; then
    echo "Model artifacts not found locally in container. Checking DVC remote..."
    if command -v dvc &> /dev/null; then
        dvc pull models/best_model.pkl.dvc models/preprocessor.pkl.dvc || true
    fi
fi

if [ ! -s "models/best_model.pkl" ] || [ ! -s "models/preprocessor.pkl" ] || [ ! -s "models/model_metadata.json" ]; then
    echo "WARNING: one or more model artifacts are missing or empty. /predict will return 503 until a trained model is present."
fi

exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"