# deep_learning_workshop

A set of files with code I use when teaching deep learning

Interesting files:
- net.py - contains implementation of a simple neural network class
- mnist.py - trains a simple network on mnist dataset, outputs result to console and `/tmp/mnist.html`
- simple_tensorflow.py - code demonstrating use of placeholders, variables and sessions in tensorflow
- mnist_tensorflow.py - trains a simple network built in tensorflow on mnist dataset, outputs result to console and `/tmp/mnist.html`

Major dependiencies are:
- python 3.5
- opencv
- tensorflow 1.3+
- numpy
- tqdm
- https://github.com/dchaplinsky/visual-logging

You can install most dependencies with conda using `conda env create -f environment.yml`. 
One caveat is `visual-logging` library - as of this writing (December 2017), its pip version is out of date, so it's best to install it directly from github with `pip install git+https://github.com/dchaplinsky/visual-logging`.

Make sure your PYTHONPATH environmental variable contains current directory, so Python can find modules you are trying to run. You can set PYTHONPATH by adding following to your `.bash_profile` (OSX) or `.bash_rc` (LINUX) scripts:  
`export PYTHONPATH="."`
