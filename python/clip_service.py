import torch
import clip
from PIL import Image
import base64
import io
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# --- Configuration ---
# Check for CUDA (GPU) availability. This is crucial for performance.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("Warning: CUDA not available. CLIP is running on CPU, which will be very slow.")

# Load the CLIP model into memory *once* when the server starts.
# 'ViT-B/32' is a good balance of speed and accuracy.
print(f"Loading CLIP model 'ViT-B/32' onto device: {DEVICE}...")
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
print("Model loaded successfully.")

# --- API Data Models ---
class CLIPRequest(BaseModel):
    # The image frame, sent as a Base64-encoded string
    image_base64: str
    # The list of text prompts to compare against
    prompts: List[str]

class CLIPResponse(BaseModel):
    # A dictionary of scores, e.g., {"a photo of a dog": 0.9, "a photo of a cat": 0.1}
    scores: Dict[str, float]

# --- FastAPI Application ---
app = FastAPI(
    title="Self-Hosted CLIP Analyser Service",
    description="An API to analyse image frames against text prompts using OpenAI's CLIP model."
)

@app.get("/")
def read_root():
    return {"status": "CLIP Analyser Service is running.", "device": DEVICE}

@app.post("/analyse_frame", response_model=CLIPResponse)
async def analyse_frame(request: CLIPRequest):
    """
    Receives a Base64-encoded image and a list of text prompts.
    Returns the similarity scores for each prompt.
    """
    if not request.image_base64 or not request.prompts:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' or 'prompts' in request.")

    try:
        # --- Image Processing ---
        # 1. Decode the Base64 string into bytes
        image_bytes = base64.b64decode(request.image_base64)
        
        # 2. Open the image bytes using PIL (Python Imaging Library)
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # 3. Apply the CLIP-specific preprocessing
        image_input = PREPROCESS(image_pil).unsqueeze(0).to(DEVICE)
        
        # --- Text Processing ---
        # 1. Tokenize the text prompts
        text_inputs = clip.tokenize(request.prompts).to(DEVICE)

        # --- Run CLIP Model ---
        # Use torch.no_grad() for inference as we don't need to calculate gradients
        with torch.no_grad():
            # Calculate features for both image and text
            image_features = MODEL.encode_image(image_input)
            text_features = MODEL.encode_text(text_inputs)
            
            # Calculate similarity
            # This gives logits (raw scores). We apply softmax to get probabilities.
            logits_per_image, logits_per_text = MODEL(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # --- Format Response ---
        # Create a clean dictionary matching prompts to their scores
        scores_dict = {prompt: prob for prompt, prob in zip(request.prompts, probs[0])}
        
        return CLIPResponse(scores=scores_dict)

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- How to Run This File ---
# 1. Install dependencies:
#    pip install fastapi uvicorn torch clip-openai Pillow
#
# 2. On your GPU server, run this command in the terminal:
#    uvicorn clip_service:app --host 0.0.0.0 --port 8000
#
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
