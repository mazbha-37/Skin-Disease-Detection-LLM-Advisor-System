import os
import tempfile

from fastapi import HTTPException
from ultralytics import YOLO

DISEASE_CLASSES = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Benign Keratosis-like Lesions",
    "Psoriasis",
    "Seborrheic Keratoses",
    "Tinea Ringworm Candidiasis",
    "Warts Molluscum and other Viral Infections",
]


class SkinClassifier:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Running without model.")
            self.model = None
            return
        self.model = YOLO(model_path)
        print("Model loaded successfully")

    def predict(self, image_bytes: bytes) -> dict:
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            results = self.model(tmp_path, verbose=False)
            probs = results[0].probs

            top1_index = probs.top1
            top1_confidence = float(probs.top1conf)
            disease_name = DISEASE_CLASSES[top1_index]

            top5_indices = probs.top5[:3]
            top5_confs = probs.top5conf[:3].tolist()

            top_predictions = [
                {"label": DISEASE_CLASSES[idx], "confidence": round(float(conf), 4)}
                for idx, conf in zip(top5_indices, top5_confs)
            ]

            return {
                "disease": disease_name,
                "confidence": round(top1_confidence, 4),
                "top_predictions": top_predictions,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
