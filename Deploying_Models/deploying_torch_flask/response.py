import requests

resp = requests.post("http://localhost:8000/predict",
                     files={"file": open('car.jpg','rb')})