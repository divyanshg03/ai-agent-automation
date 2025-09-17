# fastapi_app.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class MedicalReportRequest(BaseModel):
    medical_report: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as file:
        return HTMLResponse(file.read())

@app.post("/api/analyze")
async def analyze_report(request: MedicalReportRequest):
    # Your existing analysis logic here
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)