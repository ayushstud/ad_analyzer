"""
Image analysis module using the BLIP vision-language model.

Generates captions and computes heuristic scores for color,
visual clarity, text density, and emotional impact.
"""

import numpy as np
import torch
from PIL import Image, ImageStat
from transformers import BlipForConditionalGeneration, BlipProcessor

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
).to(_DEVICE)


def analyze_image(image_file) -> dict:
    """Analyze an uploaded ad image and return scores + caption.

    Returns
    -------
    dict with keys:
        caption, color_score, visual_clarity_score,
        text_density_score, emotional_impact_score   (each 1-10)
    """
    image = Image.open(image_file).convert("RGB")

    brightness = ImageStat.Stat(image.convert("L")).mean[0]
    contrast = ImageStat.Stat(image.convert("L")).stddev[0]

    color_score = int(np.clip(brightness / 255 * 10, 1, 10))
    visual_clarity_score = int(np.clip(contrast / 128 * 10, 1, 10))

    inputs = _processor(images=image, return_tensors="pt").to(_DEVICE)
    output_ids = _blip_model.generate(**inputs)
    caption = _processor.decode(output_ids[0], skip_special_tokens=True)

    caption_lower = caption.lower()

    text_keywords = {"text", "sign", "writing", "words", "poster", "logo", "letter"}
    text_density_score = 8 if any(w in caption_lower for w in text_keywords) else 5

    positive_keywords = {
        "smile", "happy", "laugh", "people", "person",
        "exciting", "bright", "vibrant", "colorful",
    }
    negative_keywords = {"dark", "empty", "boring", "bland", "dull"}

    if any(w in caption_lower for w in positive_keywords):
        emotional_impact_score = 9
    elif any(w in caption_lower for w in negative_keywords):
        emotional_impact_score = 3
    else:
        emotional_impact_score = 6

    return {
        "caption": caption,
        "color_score": color_score,
        "visual_clarity_score": visual_clarity_score,
        "text_density_score": text_density_score,
        "emotional_impact_score": emotional_impact_score,
    }
