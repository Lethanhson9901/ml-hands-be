import uvicorn
from fastapi import FastAPI, File, UploadFile
import sys
import shutil
from fastapi.responses import FileResponse
from yolo_opencv import *
from starlette.responses import RedirectResponse
from pydantic import BaseModel
import base64
from fastapi.middleware.cors import CORSMiddleware
from segment.finaltest import *
APP_DESC = "smth"

app = FastAPI(title='Hand Detection', description=APP_DESC)

origins = [
    "http://localhost",
    "http://localhost:3001",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

class ODRequest(BaseModel):
    image: str
    #model: str
class ODResponse(BaseModel):
    res_image: str

class ODSegResponse(BaseModel):
    res_image: str
    class_segmentation:str

@app.post("/predict", response_model = ODResponse)
def predict(request: ODRequest):
    x = request.image.split(",")

    image_np = base64.b64decode(x[1])
    filename = 'process_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(image_np)
    res = detect_hand_yolo(filename)
    with open(res, "rb") as image_file:
        res_64 = base64.b64encode(image_file.read())
        #res_64 = str(x[0])+","+str(res_64).encode('utf-8')
        # res_64.encode('utf-8')
    return ODResponse(res_image = res_64)

@app.post("/segment", response_model = ODSegResponse)
def predict_segment(request: ODRequest):
    x = request.image.split(",")

    image_np = base64.b64decode(x[1])
    filename = 'process_segment_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(image_np)
    res, class_seg = segment(filename)
    with open(res, "rb") as image_file:
        res_64 = base64.b64encode(image_file.read())
        #res_64 = str(x[0])+","+str(res_64).encode('utf-8')
        # res_64.encode('utf-8')
    return ODSegResponse(res_image = res_64, class_segmentation = class_seg)

@app.get("/test", responses = {200: {"description":"image after run model"}})
async def cat():
    #path = "root"
    file_path = "input/Front_20201220_100910.jpg"
    #file_path = os.path.join(path, file_path)
    if os.path.exists(file_path, ):
        res = detect_hand_yolo(file_path)
        return FileResponse(res, media_type = "image")
    return {"error":"file not found"}

# async def predict_api(file: UploadFile = File(...)):
#     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     if not extension:
#         return "Image must be jpg or png format!"
#     image = read_imagefile(await file.read())
#     #prediction = detect_hand_yolo(image)
#     print(type(image), image)
#     return image

#test = predict_segment('process_image.jpg')

if __name__ == "__main__":
    uvicorn.run(app, port = 6000, host='127.0.0.1')
#uvicorn main:app --reload
