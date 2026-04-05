import os

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_URL", "http://localhost:8000")


def analyze_image(image):
    if image is None:
        return "Please upload an image", "", "", ""

    try:
        with open(image, "rb") as f:
            image_bytes = f.read()

        filename = os.path.basename(image)
        mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"

        response = httpx.post(
            f"{API_BASE}/analyze_skin",
            files={"image": (filename, image_bytes, mime)},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        disease_result = f"🔍 {data['disease']} ({data['confidence']:.1%} confidence)"
        return (
            disease_result,
            data.get("recommendations", ""),
            data.get("next_steps", ""),
            data.get("tips", ""),
        )

    except Exception as e:
        return f"Error: {str(e)}", "", "", ""


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("#  Skin Disease Detection & AI Advisor")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Skin Image")
            analyze_btn = gr.Button("Analyze", variant="primary")
            gr.Markdown(
                "**Only Supported:** Eczema, Melanoma, Atopic Dermatitis, "
                "Basal Cell Carcinoma, Melanocytic Nevi, Benign Keratosis, "
                "Psoriasis, Seborrheic Keratoses, Tinea Ringworm, Warts"
            )

        with gr.Column():
            result_box = gr.Textbox(label="Detection Result", interactive=False)
            rec_box = gr.Textbox(label="Recommendations", lines=4, interactive=False)
            next_box = gr.Textbox(label="Next Steps", lines=3, interactive=False)
            tips_box = gr.Textbox(label="Daily Tips", lines=3, interactive=False)


    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[result_box, rec_box, next_box, tips_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
