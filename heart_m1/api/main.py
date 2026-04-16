from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from heart_m1.config import HeartConfig
from heart_m1.service import HeartRiskService


class PredictRequest(BaseModel):
    csv_path: str
    output_csv_path: Optional[str] = None


app = FastAPI(title="Heart Risk M1 Service")


def _project_root() -> Path:
    # .../heart_m1/heart_m1/api/main.py -> project root is two levels up.
    return Path(__file__).resolve().parents[2]


_MODEL_DIR = _project_root() / "models"
_CFG = HeartConfig()

_service: Optional[HeartRiskService] = None


@app.on_event("startup")
def _startup() -> None:
    global _service
    _service = HeartRiskService(cfg=_CFG, model_dir=_MODEL_DIR)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Heart Risk M1</title>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 32px; }
          .card { max-width: 840px; border: 1px solid #e6e6e6; border-radius: 12px; padding: 20px; }
          .row { margin: 14px 0; }
          code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
          button { padding: 10px 14px; border-radius: 10px; border: 1px solid #ddd; background: #111; color: #fff; cursor: pointer; }
          button.secondary { background: #fff; color: #111; }
          input[type="text"] { width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #ddd; }
          pre { white-space: pre-wrap; word-break: break-word; background: #0b1020; color: #e8e8e8; padding: 14px; border-radius: 12px; }
          .muted { color: #666; }
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Heart Risk M1</h2>
          <p class="muted">Загрузите CSV (как <code>heart_test.csv</code>) и получите предсказания.</p>

          <div class="row">
            <form id="uploadForm">
              <label><b>CSV файл</b></label><br/>
              <input type="file" id="file" name="file" accept=".csv" required />
              <div style="height:10px"></div>
              <label><b>Сохранить predictions.csv (опционально)</b></label><br/>
              <input type="text" id="output_csv_path" name="output_csv_path"
                     placeholder="outputs/student_predictions.csv" />
              <div style="height:14px"></div>
              <button type="submit">Predict</button>
              <button class="secondary" type="button" id="downloadBtn" disabled>Download predictions.csv</button>
            </form>
          </div>

          <div class="row">
            <p class="muted">Также доступно API: <code>POST /predict</code> и Swagger: <code>/docs</code></p>
          </div>

          <div class="row">
            <pre id="result">Готово. Выберите CSV и нажмите Predict.</pre>
          </div>
        </div>

        <script>
          const form = document.getElementById("uploadForm");
          const result = document.getElementById("result");
          const downloadBtn = document.getElementById("downloadBtn");
          let lastCsvText = null;

          function toCsv(predictions) {
            // Match test.py expectations: it reads with index_col=0, so include an index column.
            let lines = ["idx,id,prediction"];
            for (let i = 0; i < predictions.length; i++) {
              const row = predictions[i];
              lines.push(`${i},${row.id},${row.prediction}`);
            }
            return lines.join("\\n") + "\\n";
          }

          downloadBtn.addEventListener("click", () => {
            if (!lastCsvText) return;
            const blob = new Blob([lastCsvText], { type: "text/csv;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "predictions.csv";
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
          });

          form.addEventListener("submit", async (e) => {
            e.preventDefault();
            result.textContent = "Считаю предсказания...";
            downloadBtn.disabled = true;
            lastCsvText = null;

            const fileInput = document.getElementById("file");
            const outPath = document.getElementById("output_csv_path").value;
            const fd = new FormData();
            fd.append("file", fileInput.files[0]);
            if (outPath) fd.append("output_csv_path", outPath);

            try {
              const resp = await fetch("/predict_upload", { method: "POST", body: fd });
              const data = await resp.json();
              if (!resp.ok) {
                result.textContent = "Ошибка: " + JSON.stringify(data, null, 2);
                return;
              }
              result.textContent = JSON.stringify(data, null, 2);
              if (data && data.predictions) {
                lastCsvText = toCsv(data.predictions);
                downloadBtn.disabled = false;
              }
            } catch (err) {
              result.textContent = "Ошибка запроса: " + err;
            }
          });
        </script>
      </body>
    </html>
    """


@app.post("/predict_upload")
async def predict_upload(
    file: UploadFile = File(...),
    output_csv_path: Optional[str] = Form(default=None),
) -> JSONResponse:
    if _service is None:
        raise HTTPException(status_code=500, detail="Service is not initialized")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        import pandas as pd
        from io import BytesIO

        content = await file.read()
        df = pd.read_csv(BytesIO(content))

        # If requested, write predictions.csv to the given path.
        if output_csv_path:
            out_path = Path(output_csv_path)
            # We'll generate via existing service method by writing a temp file in workspace.
            tmp_dir = _project_root() / "tmp_uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_csv = tmp_dir / "uploaded.csv"
            tmp_csv.write_bytes(content)

            _service.predict_csv(test_csv_path=tmp_csv, output_csv_path=out_path)

        data = _service.predict_json_from_df(df)
        return JSONResponse(content=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if _service is None:
        raise HTTPException(status_code=500, detail="Service is not initialized")

    csv_path = Path(req.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {csv_path}")

    try:
        if req.output_csv_path:
            out_path = Path(req.output_csv_path)
            _service.predict_csv(test_csv_path=csv_path, output_csv_path=out_path)
            # Still return JSON for convenience.
            return _service.predict_json(test_csv_path=csv_path)

        return _service.predict_json(test_csv_path=csv_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

