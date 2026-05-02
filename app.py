"""
AdLens - AI-powered ad analysis Flask application.

Combines a trained ML ensemble with an optional BLIP vision-language
model to score ads and generate actionable suggestions.
"""

import os

import numpy as np
from flask import Flask, render_template, request

from ml_engine import predictor

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _score_color(score: int) -> tuple[str, str, str]:
    """Return (feedback_text, color_hex, color_glow) for a given score."""
    if score >= 80:
        return ("Excellent Ad Potential", "#ccff00", "rgba(204, 255, 0, 0.5)")
    if score >= 60:
        return ("Good, But Could Improve", "#facc15", "rgba(250, 204, 21, 0.5)")
    return ("Needs Optimization", "#f87171", "rgba(248, 113, 113, 0.5)")


def _build_suggestions(
    platform: int,
    ad_type: int,
    audience: int,
    day: int,
    hour: int,
    vlm_results: dict | None,
) -> tuple[list[str], list[str], int]:
    """Generate suggestions and insights; return (suggestions, insights, score_adjustment)."""
    suggestions: list[str] = []
    insights: list[str] = []
    adj = 0

    if platform == 1 and ad_type != 2:
        suggestions.append("Consider turning this into a video/Reel for Instagram.")
        insights.append("Video outperforms static images heavily on Instagram (+15% potential).")

    if audience == 2:  # B2B
        if platform == 1:
            suggestions.append("Run this on LinkedIn/Facebook instead for targeted B2B.")
            insights.append("Instagram has lower engagement for B2B audiences (-10%).")
            adj -= 10
        if day in (6, 7):
            suggestions.append("B2B audiences check out on weekends. Reschedule to Tue-Thu.")
            insights.append("Publishing B2B content on weekends drops reach (-15%).")
            adj -= 10
    elif audience in (1, 3):
        if day in (6, 7):
            insights.append("Weekends are peak activity for General/Student audiences (+10%).")
            adj += 8

    if hour < 7 or 13 < hour < 17:
        suggestions.append("Reschedule post for evening peak times (18:00-21:00).")
        insights.append(f"Posting at {hour}:00 hurts initial algorithmic reach (-8%).")
        adj -= 5

    if vlm_results:
        insights.append(f"Visual AI detected: '{vlm_results['caption'].capitalize()}'")
        insights.append(f"Visual Clarity Score: {vlm_results['visual_clarity_score']}/10")
        insights.append(f"Emotional Impact Score: {vlm_results['emotional_impact_score']}/10")

        if vlm_results["visual_clarity_score"] < 5:
            suggestions.append("Upload a higher-contrast/clearer image to make main elements pop.")
        if vlm_results["emotional_impact_score"] < 5:
            suggestions.append(
                "Image feels a bit plain. Add human faces or vibrant colors to boost engagement."
            )
        if vlm_results["text_density_score"] >= 8:
            suggestions.append("Image contains heavy text. Platforms may reject ads with too much text.")

    return suggestions, insights, adj


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html", ad_type=1)

    # --- Parse form inputs ---
    ad_type = int(request.form.get("ad_type", 1))
    color_val = int(request.form.get("color", 6))
    audience = int(request.form.get("audience", 3))
    platform = int(request.form.get("platform", 1))
    day = int(request.form.get("day", 3))
    hour = int(request.form.get("hour", 18))
    headline = request.form.get("headline", "").strip()
    headline_length = max(1, len(headline.split()))

    # --- ML prediction ---
    ml_score = predictor.predict(
        headline_length=headline_length,
        ad_type=ad_type,
        color_score=color_val,
        audience=audience,
        platform=platform,
        posting_hour=hour,
        posting_day=day,
    )

    score = ml_score
    vlm_score = None
    vlm_results = None

    # --- Optional VLM image analysis ---
    uploaded = request.files.get("ad_image")
    if uploaded and uploaded.filename:
        try:
            from image_analyzer import analyze_image

            vlm_results = analyze_image(uploaded)
            vlm_score = int(
                np.mean([
                    vlm_results["color_score"],
                    vlm_results["visual_clarity_score"],
                    10 - vlm_results["text_density_score"],
                    vlm_results["emotional_impact_score"],
                ])
                * 10
            )
            score = int(0.6 * ml_score + 0.4 * vlm_score)
        except Exception as exc:
            print(f"Image analysis failed: {exc}")

    # --- Suggestions & insights ---
    suggestions, insights, adj = _build_suggestions(
        platform, ad_type, audience, day, hour, vlm_results,
    )
    score = int(np.clip(score + adj, 0, 100))

    feedback, color_hex, color_glow = _score_color(score)

    return render_template(
        "index.html",
        score=score,
        ml_score=ml_score,
        vlm_score=vlm_score,
        feedback=feedback,
        suggestions=suggestions,
        insights=insights,
        color_hex=color_hex,
        color_glow=color_glow,
        ad_type=ad_type,
        color_val=color_val,
        audience=audience,
        platform=platform,
        day=day,
        hour=hour,
        headline=headline,
        cv_results=predictor.cv_results,
        best_model=predictor.best_model_name,
    )


if __name__ == "__main__":
    app.run(debug=True)
