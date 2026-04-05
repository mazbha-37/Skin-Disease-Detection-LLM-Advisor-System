import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.classifier import SkinClassifier
from app.database import init_db, log_analysis
from app.llm import get_recommendations
from app.schemas import AnalysisResponse, HealthResponse

load_dotenv()

app = FastAPI(
    title="Skin Disease Detection API",
    description="Upload skin images to detect diseases using YOLOv8 + Gemini AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    app.state.classifier = SkinClassifier("checkpoints/best.pt")
    print("API ready")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=app.state.classifier.model is not None,
    )


@app.post("/analyze_skin", response_model=AnalysisResponse)
async def analyze_skin(image: UploadFile):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted")

    image_bytes = await image.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")

    start = time.time()

    result = app.state.classifier.predict(image_bytes)
    llm_result = get_recommendations(result["disease"], result["confidence"])

    processing_time_ms = (time.time() - start) * 1000
    print(f"Processing time: {processing_time_ms:.1f}ms")

    try:
        log_analysis(
            disease=result["disease"],
            confidence=result["confidence"],
            llm_used=True,
            processing_time_ms=processing_time_ms,
        )
    except Exception as e:
        print(f"DB logging error: {e}")

    return AnalysisResponse(
        disease=result["disease"],
        confidence=result["confidence"],
        recommendations=llm_result["recommendations"],
        next_steps=llm_result["next_steps"],
        tips=llm_result["tips"],
    )
