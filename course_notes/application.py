"""
Module creating flask application
"""

import flask


def get_app():

    app = flask.Flask("deep_learning_workshop")

    @app.route("/deeplearningworkshop.html")
    def render_index_page():
        return flask.render_template("index.html")

    @app.route("/convolutional_layers_field_of_view.html")
    def render_convolutional_layers_field_of_view_page():
        return flask.render_template("convolutional_layers_field_of_view/index.html")

    return app
