from flask_bootstrap import Bootstrap
from flask import Flask

bootstrap = Bootstrap()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'sadad131[]'
    bootstrap.init_app(app)

    return app