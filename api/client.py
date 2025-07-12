import requests
from PIL import Image
import io
import base64
from pathlib import Path

with open(Path(__file__).parent/"demo.png", "rb") as f:
    image_bytes = f.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

payload = {
    "data": [
        "A metallic rusted surface",
        {"name": "image.png", "data": base64_image},
        20,
        0,
        True
    ]
}

response = requests.post("http://localhost:7860/predict", json=payload)

if response.ok:
    output_base64 = response.json()["data"][0]
    image = Image.open(io.BytesIO(base64.b64decode(output_base64)))
    image.save("out.png")
else:
    print("Error:", response.text)
