import bentoml
from bentoml.io import Multipart, JSON, Image
from PIL import Image as PILImage


model=bentoml.pytorch.get("model:latest")
transform = model.custom_objects['test_transformation']
model_runner = model.to_runner()
svc = bentoml.Service("image_classifier_service", runners=[model_runner])

@svc.api(input=Multipart(image=Image()), output=JSON())
async def predict(image: PILImage.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transformed_image = transform(image)

    batch_tensor = transformed_image.unsqueeze(0)

    outputs = await svc.runners[0].async_run(batch_tensor)


    predicted = outputs.argmax(dim=-1)
    return predicted.numpy().tolist()

# use when the file equal src in dir tree
# bentoml serve service.py:svc --reload
# bentoml build
# bentoml containerize image_classifier_service:latest -t endpoint
# bentoml delete image_classifier_service --yes
# bentoml models delete model --yes