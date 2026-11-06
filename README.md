# Heart Disease Prediction API

FastAPI service serving a scikit-learn classifier (RandomForest) trained on the
[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
Dockerized and deployable to Render.

## Project structure
```
app/            FastAPI app (main.py, schemas.py)
model/          heart_model.joblib (trained model, committed to git)
train.py        training script
Dockerfile
docker-compose.yml
requirements.txt
```

## 1. Train the model
Download `heart.csv` from Kaggle into `data/heart.csv`, then:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train.py
```
This writes `model/heart_model.joblib`. Train with the pinned
`requirements.txt` so the scikit-learn version matches the Docker image.

## 2. Run with Docker
```bash
docker-compose build
docker-compose up
```
Swagger UI: http://localhost:8000/docs

## 3. Endpoints
| Method | Path     | Description                          |
|--------|----------|--------------------------------------|
| GET    | /health  | Liveness check                       |
| GET    | /info    | Model type and feature list          |
| POST   | /predict | Returns `{"heart_disease": true/false}` |

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,
       "thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

## 4. Deploy on Render
5. Deploy. Live URL:(https://dashboard.render.com/web/new)


