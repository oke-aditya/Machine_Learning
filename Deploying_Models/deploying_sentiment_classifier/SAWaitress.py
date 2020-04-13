from waitress import serve
import flask_adv_deploy

serve(flask_adv_deploy.app, port=8000, threads=6)
