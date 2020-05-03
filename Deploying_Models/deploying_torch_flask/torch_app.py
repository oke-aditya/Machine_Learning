from flask import Flask, jsonify, request
from torchvision import models
import json
import io
import torchvision.transforms as transforms
from PIL import Image


app = Flask(__name__)


# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()

imagenet_class_index = json.load(open('imagenet_class_index.json'))


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        print(class_id, class_name)

    return jsonify({'class_id': class_id, 'class_name': class_name})

# Did you notice that model variable is not part of get_prediction method? 
# Or why is model a global variable? Loading a model can be an expensive operation in terms of memory and compute. 
# If we loaded the model in the get_prediction method, then it would get unnecessarily loaded every time the method is called. 
# Since, we are building a web server, there could be thousands of requests per second, 
# we should not waste time redundantly loading the model for every inference. 
# So, we keep the model loaded in memory just once. 
# In production systems, it’s necessary to be efficient about your use of compute to be able to serve requests at scale, 
# so you should generally load your model before serving requests.


if __name__ == '__main__':
    app.run(debug=True)

    