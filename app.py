import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("best_bone_fracture_model.h5")

img_size = 224

def preprocess_image(image):
    image = image.resize((img_size, img_size))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return{"message": "Bone Fracture Detection API"}

class_names = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

@app.post("/predict/", summary = "Predict bone fracture type")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    processed = preprocess_image(image)
    prediction = model.predict(processed)

    prediction = prediction[0]

    prediction_index = np.argmax(prediction)
    prediction_label = class_names[prediction_index]
    confidence = float(prediction[prediction_index])

    return {"prediction_class": prediction_label,
            "confidence": round(confidence, 4)
            }
