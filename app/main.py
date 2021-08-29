from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi.templating import Jinja2Templates
import os
from PIL import Image
import io
import sys
import logging

from prediction_response import PredictionResponseDto
from prediction import ImageClassifier

templates = Jinja2Templates(directory="templates")

app = FastAPI()

image_classifier = ImageClassifier()

@app.get("/", response_class=HTMLResponse)
async def upload_file(request: Request):
    return templates.TemplateResponse("index.html",{"request": request}) 


@app.post("/",response_class=HTMLResponse)
async def upload_image_file(request: Request, file: UploadFile = File(...)):    
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')    

    try:
        contents = await file.read()   
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        result = image_classifier.predict(image)
        logging.info(f"Predicted Class:     {result[0]}")
        result_str = f"Predicted Label:     {result[0]}"
        result_str2 = f"Confidence:     {str(result[1])}"
        return templates.TemplateResponse("index.html",{"request": request, "result":result_str, "result2":result_str2})
    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))