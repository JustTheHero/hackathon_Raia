from fastapi import FastAPI
from fastapi.responses import FileResponse
from models.isso import test_enhanced_model

app = FastAPI()

API_KEY = ""

@app.get("/predict")
def run_prediction():
    report_filename = test_enhanced_model(openai_api_key=API_KEY)

    return FileResponse(
        report_filename,
        media_type="text/plain",
        filename="report.txt"
    )
