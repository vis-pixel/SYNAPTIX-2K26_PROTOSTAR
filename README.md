# BioRhythm X — Research-Grade AI Backend

> **Wearable Health + Nutrition Intelligence Platform**
> Python FastAPI · PostgreSQL · Redis · WebSockets · JWT · Docker · ML (PyTorch + scikit-learn)

---

## 🗂️ Project Structure

```
D:\1.0\
├── main.py                          # FastAPI entry point
├── .env                             # Environment config
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── app/
│   ├── config.py                    # Settings (pydantic-settings)
│   ├── database.py                  # Async SQLAlchemy engine + session
│   ├── logging_config.py            # Loguru structured logging
│   │
│   ├── models/                      # SQLAlchemy ORM models
│   │   ├── user.py                  # User + BaselineProfile
│   │   ├── vitals.py                # LiveVitals
│   │   ├── step_metrics.py          # StepMetrics
│   │   ├── calorie_metrics.py       # CalorieMetrics
│   │   ├── diet.py                  # DietPlan + MealPlan + MacroTargets
│   │   ├── anomaly.py               # AnomalyLog
│   │   └── ml_models.py             # MLModelVersion + DatasetRecord
│   │
│   ├── routes/                      # FastAPI routers
│   │   ├── auth.py                  # Register, Login, Refresh, /me
│   │   ├── websocket.py             # ws://…/ws/vitals/{user_id}
│   │   ├── vitals.py                # POST /ingest, GET /history
│   │   ├── steps.py                 # POST /analyze
│   │   ├── calories.py              # POST /estimate
│   │   ├── diet.py                  # POST /plan, /adapt, GET /food-db
│   │   ├── predictions.py           # POST /analyze
│   │   ├── anomaly.py               # POST /detect, GET /logs/{user_id}
│   │   ├── risk.py                  # POST /score
│   │   └── datasets.py              # GET /list, /status, POST /download, /upload
│   │
│   ├── step_engine/
│   │   └── step_service.py          # Peak detection, gait, cadence, activity
│   │
│   ├── calorie_engine/
│   │   └── calorie_service.py       # BMR, TDEE, VO2, fat/carb burn, fatigue
│   │
│   ├── diet_engine/
│   │   ├── macro_calculator.py      # Macros, carb cycling
│   │   ├── meal_planner.py          # 7-day planner + adaptive engine
│   │   └── food_database.py         # 25+ foods with macros + micronutrients
│   │
│   ├── ml/
│   │   └── models.py                # IsolationForest + LSTM + Autoencoder
│   │
│   ├── anomaly_engine/
│   │   └── detector.py              # Ensemble anomaly detection
│   │
│   ├── dataset_loader/
│   │   ├── registry.py              # Dataset configs (MIT-BIH, WESAD, etc.)
│   │   ├── downloader.py            # Async background downloader
│   │   ├── parser.py                # WFDB/EDF/CSV/WESAD parsers
│   │   └── normalizer.py            # Unified internal schema mapper
│   │
│   ├── services/
│   │   ├── auth_service.py          # JWT + bcrypt
│   │   └── prediction_service.py    # 8-dimensional biometric predictor
│   │
│   └── synthetic_generator/
│       └── generator.py             # Synthetic wearable data
│
├── scripts/
│   ├── train_models.py              # Train all ML models
│   ├── simulate_wearable.py         # WebSocket simulator
│   ├── test_diet_plan.py            # Diet pipeline tester
│   └── simulate_stress.py           # Stress scenario demo
│
├── datasets/                        # Downloaded research datasets
│   ├── mit_bih/
│   ├── fantasia/
│   ├── bidmc/
│   ├── sleep_edf/
│   ├── wesad/
│   └── mhealth/
│
└── ml_models/                       # Saved trained models
    ├── isolation_forest.pkl
    ├── lstm_model.pth
    └── autoencoder.pth
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose
- (Optional) CUDA GPU for faster ML training

### 2. Clone & Configure

```bash
cd D:\1.0
# Copy the env file (already pre-filled with defaults)
copy .env .env.local
```

### 3. Start Services (PostgreSQL + Redis)

```bash
docker-compose up -d db redis
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

### 6. Full Docker Deploy (all services)

```bash
docker-compose up --build
```

---

## 🧪 Testing & Scripts

### Train ML Models (Synthetic Data — No Dataset Required)
```bash
python scripts/train_models.py --synthetic --samples 10000
```

### Train on a Real Dataset
```bash
# First download the dataset
curl -X POST "http://localhost:8000/api/datasets/download?name=mit_bih" \
  -H "Authorization: Bearer <TOKEN>"

# Then train
python scripts/train_models.py --dataset mit_bih --model isolation_forest
```

### Test Diet Plan Generation (No Server Required)
```bash
python scripts/test_diet_plan.py
```

### Simulate Stress Scenario (No Server Required)
```bash
python scripts/simulate_stress.py
```

### Simulate Wearable (Requires Running Server + Auth Token)
```bash
# 1. Register a user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@bio.com","username":"testuser","password":"password123"}'

# 2. Login to get token
curl -X POST http://localhost:8000/api/auth/login \
  -d "username=test@bio.com&password=password123"

# 3. Run wearable simulator
python scripts/simulate_wearable.py \
  --user-id <UUID-FROM-LOGIN> \
  --token <ACCESS-TOKEN> \
  --activity Run \
  --samples 200
```

---

## 📡 Key API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/auth/register` | Register user |
| POST | `/api/auth/login` | Login → JWT tokens |
| `WS` | `/ws/vitals/{user_id}` | Real-time wearable stream |
| POST | `/api/vitals/ingest` | HTTP vitals ingestion |
| POST | `/api/steps/analyze` | Gait + step analysis |
| POST | `/api/calories/estimate` | Calorie burn estimation |
| POST | `/api/diet/plan` | Generate 7-day diet plan |
| POST | `/api/diet/adapt` | Adaptive diet (HRV/stress/sleep) |
| GET | `/api/diet/food-db` | Browse food database |
| POST | `/api/predictions/analyze` | 8-dimensional biometric predictions |
| POST | `/api/anomaly/detect` | Real-time anomaly detection |
| GET | `/api/anomaly/logs/{user_id}` | Anomaly history |
| POST | `/api/risk/score` | Composite AI risk score |
| GET | `/api/datasets/list` | List all datasets |
| GET | `/api/datasets/status` | Dataset download status |
| POST | `/api/datasets/download?name=mit_bih` | Background download |
| POST | `/api/datasets/upload?name=wesad` | Manual dataset upload |

---

## 🤖 ML Models

| Model | Type | Use Case |
|-------|------|----------|
| **IsolationForest** | sklearn | Real-time anomaly scoring (per-sample) |
| **LSTM** | PyTorch | Sequence-based trend prediction |
| **Autoencoder** | PyTorch | Reconstruction-based anomaly (adaptive threshold) |

**Ensemble strategy**: `0.40×IsolationForest + 0.35×Autoencoder + 0.25×ThresholdCheck`

---

## 📊 Supported Datasets

| Dataset | Source | Signals |
|---------|--------|---------|
| **MIT-BIH Arrhythmia** | PhysioNet | ECG |
| **Fantasia** | PhysioNet | ECG, HRV |
| **BIDMC PPG** | PhysioNet | PPG, Respiration, ECG |
| **Sleep-EDF** | PhysioNet | EEG, Sleep Stages |
| **WESAD** | Uni Siegen | ECG, EDA, Stress Labels |
| **MHEALTH** | UCI | Accelerometer, Activity Labels |

**Dataset Mode** (`DATASET_MODE` in `.env`):
- `auto` — automatic background download
- `manual` — upload via `/api/datasets/upload`

---

## 🧬 Diet Intelligence Modes

| Mode | Key Features |
|------|-------------|
| **Gym Rats** | High protein, carb cycling, creatine, post-workout recovery |
| **Healthy Human** | Balanced macros, fiber, anti-inflammatory foods |
| **Fat Loss** | Calorie deficit, high satiety, metabolic adaptation |
| **Indian** | Roti, dal, paneer, regional foods |
| **Vegan/Vegetarian** | Plant-based complete proteins, B12 flagging |
| **Low Carb** | <100g carbs, high fat, electrolyte emphasis |

**Adaptive Diet**: Automatically adjusts macros if HRV drops, stress is high, sleep is poor, fatigue is elevated, or illness risk rises.

---

## 🔐 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | PostgreSQL | Async SQLAlchemy URL |
| `REDIS_URL` | redis://localhost:6379 | Redis connection |
| `JWT_SECRET_KEY` | `change-me` | JWT signing key |
| `DATASET_MODE` | `auto` | `auto` or `manual` |
| `DATASET_BASE_DIR` | `./datasets` | Dataset storage path |
| `ML_MODELS_DIR` | `./ml_models` | Trained model storage |

---

## 🏗️ Architecture

```
WebSocket/HTTP  →  FastAPI Routes
                      ↓
               Services Layer
          ┌────────────────────────┐
          │  Step Engine           │  scipy peak detection + gait
          │  Calorie Engine        │  MET + Karvonen + VO2
          │  Diet Intelligence     │  Mifflin-St Jeor + Adaptive
          │  Anomaly Engine        │  IF + AE + Threshold
          │  Prediction Service    │  8-dimensional clinical scores
          │  ML Models             │  IsolationForest + LSTM + AE
          └────────────────────────┘
                      ↓
          PostgreSQL (async) + Redis pub/sub
```
