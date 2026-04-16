# Heart Risk M1

## Что делает проект

Сервис обучает модель машинного обучения для предсказания `Heart Attack Risk (Binary)` и умеет:
1) принимать CSV с тестовыми данными (в нём нет таргета) и
2) возвращать предсказания в формате JSON и/или сохранять `predictions.csv` в нужном формате для проверки.

Проверка `test.py` ожидает CSV с колонками после `pd.read_csv(..., index_col=0)`:
`["id", "prediction"]`.

## Установка

```bash
cd heart_m1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Подготовка данных 

Положите файлы в папку `data/` рядом с `README.md`:
- `data/heart_train.csv`
- `data/heart_test.csv`

## Обучение модели

```bash
source .venv/bin/activate
python scripts/train_model.py \
  --train-csv data/heart_train.csv \
  --model-dir models
```

## Предсказания на тестовой выборке

```bash
python scripts/predict_csv.py \
  --test-csv data/heart_test.csv \
  --output outputs/student_predictions.csv \
  --model-dir models
```

## Запуск FastAPI сервиса

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8000
```

Endpoint:
`POST /predict`

Тело запроса:
```json
{ "csv_path": "data/heart_test.csv" }
```

Ответ:
```json
{
  "predictions": [
    { "id": 123, "prediction": 0 },
    { "id": 456, "prediction": 1 }
  ]
}
```

## Архитектура (ООП)

Ключевые классы:

1) `heart_m1.data.HeartCSVData`
   - строит `feature_spec` на обучающих данных
   - приводит train/test DataFrame к формату для CatBoost
2) `heart_m1.model.CatBoostHeartRiskModel`
   - обертка над `CatBoostClassifier` (fit/predict_proba/predict_labels)
3) `heart_m1.model.HeartModelRepository`
   - сохранение/загрузка `heart_catboost.cbm` и `meta.json`
4) `heart_m1.training.HeartRiskTrainer`
   - обучает модель и выбирает `threshold` по max `f1_macro` на валидации
5) `heart_m1.predicting.HeartRiskPredictor`
   - предсказывает и формирует `DataFrame(id, prediction)`
6) `heart_m1.service.HeartRiskService`
   - “фасад”: train/predict для скриптов и FastAPI

## Файлы/артефакты

После обучения в `models/` появятся:
- `heart_catboost.cbm`
- `meta.json`

