# ECG Anomaly Detection using Autoencoders

An unsupervised deep learning system that detects abnormal heartbeats
from ECG signals using a Dense Autoencoder — trained exclusively on
normal beats, requiring zero abnormal labels during training.

## Project Structure

- `app.py` — Streamlit UI
- `ecg_autoencoder.keras` — Trained autoencoder model
- `scaler.pkl` — Fitted MinMax scaler
- `threshold.pkl` — Anomaly detection threshold
- `test_ecg_mixed.csv` — Sample test data

## Tech Stack

- TensorFlow / Keras
- Streamlit
- Scikit-learn
- Plotly

## How to Run

```bash
# Create virtual environment
python -m venv ecg_env
ecg_env\Scripts\activate       # Windows
source ecg_env/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Results

- Balanced Accuracy: 72%
- ROC-AUC: 0.817
- Threshold: 80th percentile of normal reconstruction errors
