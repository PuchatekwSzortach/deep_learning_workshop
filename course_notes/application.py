"""
Module creating flask application
"""

import flask


def get_app():

    app = flask.Flask("deep_learning_workshop")

    @app.route("/deeplearningworkshop.html")
    def render_index():
        return flask.render_template("index.html")

    return app
