# Student Performance Intelligence

A CPPE mini project that estimates a student's final mathematics grade (0–20) using a machine-learning regression pipeline. It demonstrates data preprocessing, model comparison, MLflow experiment tracking, an interactive Streamlit UI, testing, and Hugging Face Spaces deployment.

## Models
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

The selected model is the model with the lowest RMSE on a held-out test split. The saved pipeline includes preprocessing and the regressor.

## Dataset
UCI Student Performance dataset, `student-mat.csv`. The training script downloads the public dataset from the UCI Machine Learning Repository. The target is `G3` (final mathematics grade). Inputs include `studytime`, `failures`, `absences`, `G1`, `G2`, `schoolsup`, `famsup`, and `internet`.

## Local setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

## MLflow UI
In a second activated terminal:
```powershell
.\.venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```
Open `http://127.0.0.1:5000`.

## Hugging Face Spaces
Create a new **Streamlit** Space. Upload `app.py`, `requirements-hf.txt` renamed to `requirements.txt`, and the two files from `models/`:
- `student_performance_model.joblib`
- `model_metadata.json`

Do not upload `mlruns/` or the raw dataset. The deployed app only needs the saved model and metadata.

## Limitations
This project is an educational demonstration. Predictions are estimates based on historical data, not guaranteed outcomes. The dataset is limited in scope and should not be used as the sole basis for real academic decisions.
