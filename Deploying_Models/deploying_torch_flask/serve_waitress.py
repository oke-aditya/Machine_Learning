from waitress import serve
import torch_app

serve(torch_app.app, port=8000, threads=6)
