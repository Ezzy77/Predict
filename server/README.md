# GoalCast API Server

FastAPI backend for football match predictions (Big 5 leagues).

## Setup

```bash
cd server
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download data (optional)

```bash
python download.py
```

## Train models

```bash
python train.py
```

## Run API

```bash
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs
