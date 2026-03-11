# FedPhishAI — Privacy-Preserving Phishing Email Detection

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
```bash
python scripts/prepare_data.py
```

### 2. Train Baseline (DistilBERT)
```bash
python scripts/run_baseline.py
```

### 3. Test Model
```bash
python scripts/test_model.py
```

### 4. Run Federated Learning
```bash
python scripts/run_federated.py
```

### 5. Plot Metrics
```bash
python scripts/plot_metrics.py
```

## Key Fix: Shortcut Learning
The model was predicting all emails as phishing because it learned URL/email presence as shortcuts.
Fixed by replacing URLs → `urltoken` and emails → `emailtoken` in `preprocessor.py`, forcing semantic learning.