import bentoml
from bentoml.io import Multipart, JSON, Image
from PIL import Image as PILImage
from torchvision import transforms


model=bentoml.pytorch.get("model:latest")
# transform = model.custom_objects['test_transformation'] # can't use in dockerize
transform =transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
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

# use when the file on the same level as src folder in directory tree
# bentoml serve service.py:svc --reload
# bentoml build
# bentoml containerize image_classifier_service:latest -t endpoint
# bentoml delete image_classifier_service --yes
# bentoml models delete model --yes