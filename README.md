# simple_\_neural_\_network

A basic implementation of a neural network.
Two files are provided:
- net.py - contains definition of a simple neural network class
- mnist.py - trains a simple network on mnist dataset, outputs result to console and `/tmp/mnist.html`

This code is inspired by textbook "Neural networks and Deep Learning" by Michael A. Nielsen, and its accompanying code samples.

You can install most dependencies with conda using `conda env create -f environment.yml`. 
One caveat is `visual-logging` library - its pip version is out of date, so it's best to install it directly from github with `pip install git+https://github.com/dchaplinsky/visual-logging`.
