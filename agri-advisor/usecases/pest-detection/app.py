from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
from model_inference import PestInference
from config import Config
from PIL import Image
import io

app = FastAPI()
pest_infer = PestInference()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file upload and returns the prediction.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)
        result = pest_infer.predict(temp_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
