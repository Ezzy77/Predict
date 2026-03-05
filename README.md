# GoalCast

Football match prediction app — Big 5 European leagues (Premier League, Bundesliga, La Liga, Serie A, Ligue 1).

## Structure

```
Predict/
├── server/          # FastAPI backend
│   ├── main.py      # API
│   ├── train.py     # Model training
│   ├── data_prep.py
│   ├── download.py
│   ├── models/
│   └── data/
└── frontend/        # Next.js app
```

## Quick start

**Backend** (from `server/`):
```bash
cd server
source .venv/bin/activate
pip install -r requirements.txt
python download.py   # optional: fetch match data
python train.py     # optional: train models (requires data)
uvicorn main:app --reload --port 8000
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_URL` to your backend URL (default: `http://localhost:8000`).
