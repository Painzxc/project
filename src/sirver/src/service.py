from PIL import Image
import io
import json
import logging
import pydantic
import numpy as np
from ultralytics import YOLO

from fastapi import FastAPI, File, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import torch
from torchvision import transforms

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

app = FastAPI()

service_config_path = "./src/configs/service_config.json"
with open(service_config_path, "r") as service_config:
    service_config_json = json.load(service_config)


class ServiceConfig(pydantic.BaseModel):
    name_of_classifier: str
    path_to_classifier: str
    name_of_detector: str
    path_to_detector: str


service_config_adapter = pydantic.TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(service_config_json)


class ServiceOutput(pydantic.BaseModel):
    xtl: int
    ytl: int
    xbr: int
    ybr: int
    class_name: str


detector = YOLO(service_config_python.path_to_detector)


def load_classifier(path_to_pth_weights, device):
    model = torch.load(path_to_pth_weights, map_location=device)
    model.eval()
    model.to(device)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = load_classifier(service_config_python.path_to_classifier, device)

class_names = ["Aircraft", "Ship"]  # Замените это на ваш список имен классов


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def health_check() -> str:
    return '{"Status" : "OK"}'


@app.post("/file/")
async def inference(image: UploadFile = UploadFile(...)):
    image_content = await image.read()
    image = Image.open(io.BytesIO(image_content))
    image = image.convert("RGB")
    transform = transforms.Resize((640, 640))
    image = transform(image)
    cv_image = np.array(image)
    logger.info(f"Принята картинка размерности: {cv_image.shape}")

    output_dict = {"objects": []}

    detector_outputs = detector(cv_image)
    for box in detector_outputs[0].boxes.xyxy:

        xtl, ytl, xbr, ybr = box.tolist()
        crop_object = cv_image[int(ytl) : int(ybr), int(xtl) : int(xbr)]
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        crop_tensor = transform(Image.fromarray(crop_object))
        crop_tensor = torch.unsqueeze(crop_tensor, 0)
        class_id = classify_image(classifier, crop_tensor)
        class_name = class_names[class_id]
        output_dict["objects"].append(
            {
                "xtl": int(xtl),
                "xbr": int(xbr),
                "ytl": int(ytl),
                "ybr": int(ybr),
                "class_name": class_name,
            }
        )

    service_output = {
        "objects": [ServiceOutput(**item) for item in output_dict["objects"]]
    }
    service_output_json = jsonable_encoder(service_output)

    return JSONResponse(content=service_output_json)


def classify_image(classifier, image):
    with torch.no_grad():
        output = classifier(image.to(device))
        _, predicted = torch.max(output, 1)
    return predicted.item()
