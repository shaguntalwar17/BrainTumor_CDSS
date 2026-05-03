import os
import cv2
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

try:
    from preprocess_engine import apply_preprocessing
except ImportError:
    raise ImportError("Error: 'preprocess_engine.py' file not found. ")

app = FastAPI(title="Brain Tumor CDSS - Phase 1 API")

@app.post("/upload-mri")
async def upload_mri(file: UploadFile = File(...)):
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="images upload .")

    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid Image")

    temp_filename = "temp_api_upload.jpg"
    cv2.imwrite(temp_filename, img)

    processed_data = apply_preprocessing(temp_filename)

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    if processed_data is not None:
    
        final_img = (processed_data * 255).astype(np.uint8)
        
        res, im_png = cv2.imencode(".png", final_img)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

    raise HTTPException(status_code=500, detail="Processing failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)