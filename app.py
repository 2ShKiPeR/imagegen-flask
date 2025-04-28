from flask import Flask
from keras.models import load_model
import numpy as np
import io
from PIL import Image
import base64

app = Flask(__name__)

generator = load_model('generator_model.h5')
noise_shape = 100

@app.route("/generate-image", methods=["POST"])
def generate_image():
    random_noise = np.random.normal(0, 1, size=(1, noise_shape))
    generated_image = generator.predict(random_noise)[0]
    generated_image = (generated_image * 255).astype(np.uint8)

    image = Image.fromarray(generated_image)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    image_bytes = buf.getvalue()

    base64_str = base64.b64encode(image_bytes).decode('utf-8')

    return {"image": base64_str}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
