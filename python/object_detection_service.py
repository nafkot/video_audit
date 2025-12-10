import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import base64
import io
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on DEVICE: {DEVICE}")

# 1. Grounding DINO (Object Detection)
# CPU OPTIMIZATION: Using 'tiny' model for speed
DINO_ID = "IDEA-Research/grounding-dino-tiny"
print(f"Loading DINO: {DINO_ID}...")
dino_processor = AutoProcessor.from_pretrained(DINO_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(DEVICE)

# 2. BLIP (Captioning)
CAPTION_ID = "Salesforce/blip-image-captioning-base"
print(f"Loading BLIP: {CAPTION_ID}...")
caption_processor = BlipProcessor.from_pretrained(CAPTION_ID)
caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_ID).to(DEVICE)

# 3. CLIP (Classification)
CLIP_ID = "openai/clip-vit-base-patch32"
print(f"Loading CLIP: {CLIP_ID}...")
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE)

# --- API Data Models ---
class ImageRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None

class ClipRequest(BaseModel):
    image_base64: str
    prompts: List[str]

class BoundingBox(BaseModel):
    box: List[int]
    score: float
    label: str

class DetectionResponse(BaseModel):
    detections: List[BoundingBox]

class CaptionResponse(BaseModel):
    description: str

class ScoreResponse(BaseModel):
    scores: Dict[str, float]

app = FastAPI(title="Unified Visual Analysis Service (CPU Optimized)")

@app.post("/find_objects", response_model=DetectionResponse)
async def find_objects(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = dino_processor(images=image_pil, text=request.prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = dino_model(**inputs)

        results = dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[image_pil.size[::-1]], threshold=0.25
        )[0]

        detections_list = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections_list.append(BoundingBox(box=[int(x) for x in box.tolist()], score=float(score), label=label))
        return DetectionResponse(detections=detections_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/describe", response_model=CaptionResponse)
async def describe_image(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text_input = request.prompt if request.prompt else None

        if text_input:
            inputs = caption_processor(image_pil, text=text_input, return_tensors="pt").to(DEVICE)
        else:
            inputs = caption_processor(image_pil, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # CPU FIX: Removed min_length to prevent empty strings on short descriptions
            out = caption_model.generate(**inputs, max_new_tokens=100)

        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        return CaptionResponse(description=caption)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyse_frame", response_model=ScoreResponse)
async def analyse_frame(request: ClipRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(text=request.prompts, images=image_pil, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
        scores_dict = {prompt: float(probs[i]) for i, prompt in enumerate(request.prompts)}
        return ScoreResponse(scores=scores_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
