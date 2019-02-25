"""
Module to run the server
"""

import glob

import application


if __name__ == "__main__":

    app = application.get_app()

    # To prevent loading model twice, set use_reloader to False
    app.run(host="0.0.0.0", use_reloader=True,
            extra_files=glob.glob("./templates/**/*.html", recursive=True))
