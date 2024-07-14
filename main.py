import uvicorn
from fastapi import FastAPI, File, UploadFile,Response
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.models import load_model
from loadimage import preprocess_data , plot_images
import numpy as np
import io
import tensorflow as tf
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model = load_model('flower_generator_color_landscape_48_epochs.h5')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://colourizephoto.netlify.app/upload"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post('/')
async def file_upload(file : UploadFile = File(...)):
    contents = await file.read()
    
    # Open the image using load_img from Keras
    image = load_img(io.BytesIO(contents), target_size=(256, 256))
    # image = image.resize((256, 256))

    
    # Convert the image to a numpy array
    image_array = preprocess_data(image)

    gen_image = model.predict(image_array)
    
    
    # plot_images(image_array, gen_image)
    print(tf.__version__)
    print(type(gen_image))
    gen_image = gen_image*127.5 + 127.5
    image_array = image_array*127.5 + 127.5
    gen_image =gen_image.astype(np.uint8)
    image_array = image_array.astype(np.uint8)
    gen_image = np.squeeze(gen_image,axis=0)
    image = Image.fromarray(gen_image)
    img_byte = io.BytesIO()
    image.save(img_byte,format='JPEG')
    img_byte.seek(0)

    return Response(content=img_byte.getvalue(),media_type="image/jpeg")
    # return {"file ": gen_image.shape}

# if __name__ == '__main__':
#     uvicorn.run(app,port=8000)
    