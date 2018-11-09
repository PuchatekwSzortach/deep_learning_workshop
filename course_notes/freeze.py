"""
Script for freezing Flask app into a set of static files
"""

import flask_frozen

import application


def main():

    app = application.get_app()
    freezer = flask_frozen.Freezer(app)

    freezer.freeze()


if __name__ == "__main__":
    main()
