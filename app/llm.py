import os
import re

from google import genai
from google.genai import types

URGENT_DISEASES = ["Melanoma", "Basal Cell Carcinoma"]


def get_recommendations(disease: str, confidence: float) -> dict:
    is_urgent = disease in URGENT_DISEASES
    urgency_note = (
        "This condition may be serious and needs urgent dermatologist evaluation within 48 hours."
        if is_urgent
        else "A routine dermatologist appointment is recommended."
    )

    prompt = f"""You are a dermatology AI assistant. A patient has been diagnosed with {disease} (confidence: {confidence:.0%}).

{urgency_note}

Please provide concise, practical advice in exactly this format:

RECOMMENDATIONS:
[2-3 sentences about the condition and general management]

NEXT_STEPS:
[1-2 sentences on when and how urgently to see a doctor]

TIPS:
[2-3 practical daily care tips]

Keep each section brief. Do not use bullet points or headers beyond the labels above.
"""

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return parse_response(response.text)

    except Exception as e:
        print(f"LLM error: {e}")
        return get_fallback(disease)


def parse_response(text: str) -> dict:
    recommendations = ""
    next_steps = ""
    tips = ""

    rec_match = re.search(r"RECOMMENDATIONS:\s*(.*?)(?=NEXT_STEPS:|$)", text, re.DOTALL)
    next_match = re.search(r"NEXT_STEPS:\s*(.*?)(?=TIPS:|$)", text, re.DOTALL)
    tips_match = re.search(r"TIPS:\s*(.*?)$", text, re.DOTALL)

    if rec_match:
        recommendations = rec_match.group(1).strip()
    if next_match:
        next_steps = next_match.group(1).strip()
    if tips_match:
        tips = tips_match.group(1).strip()

    if not recommendations:
        recommendations = text.strip()
        next_steps = "Please consult a dermatologist."
        tips = "Keep the affected area clean and moisturized."

    return {
        "recommendations": recommendations,
        "next_steps": next_steps,
        "tips": tips,
    }


def get_fallback(disease: str) -> dict:
    fallbacks = {
        "Eczema": {
            "recommendations": "Keep skin moisturized and avoid known triggers like harsh soaps and stress. Use fragrance-free products and wear soft, breathable fabrics.",
            "next_steps": "Book a routine dermatologist appointment within 2-4 weeks. They may prescribe topical corticosteroids for flare-ups.",
            "tips": "Apply moisturizer right after bathing. Avoid hot showers and scratching the affected area.",
        },
        "Melanoma": {
            "recommendations": "Melanoma is a serious form of skin cancer that requires immediate medical attention. Early detection significantly improves outcomes.",
            "next_steps": "See a dermatologist URGENTLY within 48 hours. Do not delay — early treatment is critical for melanoma.",
            "tips": "Avoid sun exposure on the affected area. Do not attempt to treat this at home.",
        },
        "Atopic Dermatitis": {
            "recommendations": "Atopic dermatitis causes chronic itchy, inflamed skin. Regular moisturizing and avoiding triggers helps manage symptoms.",
            "next_steps": "Schedule a dermatologist appointment for a proper treatment plan including prescription creams if needed.",
            "tips": "Use gentle, unscented skincare products. Keep nails short to prevent skin damage from scratching.",
        },
        "Basal Cell Carcinoma": {
            "recommendations": "Basal cell carcinoma is the most common skin cancer. It is usually treatable when caught early.",
            "next_steps": "See a dermatologist URGENTLY within 48 hours for proper diagnosis and treatment options.",
            "tips": "Protect the area from sun exposure. Use SPF 50+ sunscreen daily.",
        },
        "Melanocytic Nevi": {
            "recommendations": "Melanocytic nevi are common moles that are usually harmless. However they should be monitored for changes.",
            "next_steps": "Schedule a routine skin check with a dermatologist. Use the ABCDE rule to monitor changes.",
            "tips": "Take monthly photos of the mole to track any changes in size, shape, or color.",
        },
        "Benign Keratosis-like Lesions": {
            "recommendations": "These are non-cancerous skin growths that are very common with age. They do not require treatment unless bothersome.",
            "next_steps": "A routine dermatologist visit can confirm the diagnosis. Removal is optional and cosmetic.",
            "tips": "Avoid picking or scratching the lesion. Keep skin moisturized to reduce irritation.",
        },
        "Psoriasis": {
            "recommendations": "Psoriasis is a chronic autoimmune condition causing scaly skin patches. Stress and certain foods can trigger flare-ups.",
            "next_steps": "See a dermatologist for a treatment plan which may include topical treatments or light therapy.",
            "tips": "Moisturize daily, manage stress, and avoid alcohol and smoking which can worsen symptoms.",
        },
        "Seborrheic Keratoses": {
            "recommendations": "Seborrheic keratoses are harmless skin growths that look waxy or scaly. They are very common and not contagious.",
            "next_steps": "A routine dermatologist visit is recommended to confirm diagnosis. No treatment is necessary unless desired.",
            "tips": "Avoid irritating the area with tight clothing. Keep skin clean and moisturized.",
        },
        "Tinea Ringworm Candidiasis": {
            "recommendations": "This is a fungal infection that is treatable with antifungal medication. It can spread to others so avoid sharing personal items.",
            "next_steps": "Visit a doctor or dermatologist soon for antifungal treatment, either topical or oral depending on severity.",
            "tips": "Keep the affected area dry and clean. Wash towels and clothing frequently in hot water.",
        },
        "Warts Molluscum and other Viral Infections": {
            "recommendations": "These are viral skin infections that are generally harmless but contagious. Many resolve on their own over time.",
            "next_steps": "See a dermatologist for treatment options including topical treatments or minor procedures for removal.",
            "tips": "Avoid touching or picking at the affected area. Wash hands frequently to prevent spreading.",
        },
    }

    return fallbacks.get(
        disease,
        {
            "recommendations": "Please consult a dermatologist for proper evaluation of this condition.",
            "next_steps": "Book an appointment with a qualified dermatologist as soon as possible.",
            "tips": "Keep the affected area clean and avoid self-medicating.",
        },
    )
