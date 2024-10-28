from PIL import Image
import bentoml
import io

image = Image.open('artifacts/data_ingestion/data/train/NORMAL/IM-0003-0001.jpeg')

image_bytes = io.BytesIO()
image.save(image_bytes, format='JPEG')
image_bytes.seek(0)

with bentoml.client.SyncHTTPClient.from_url("http://localhost:3000") as client:
    response = client.call(
        "predict",
        image=image
    )

print(response)
