from flask_adv_deploy import app
# Note Gunicorn is not supported in Windows machine.

if __name__ == "__main__":
    app.run()
