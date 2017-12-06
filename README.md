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
- visual-logging (https://github.com/dchaplinsky/visual-logging)

You can use conda to install all the dependencies:
```bash
conda env create -f environment.yaml
source activate deep_learning_workshop
pip install pip install --upgrade git+https://github.com/dchaplinsky/visual-logging
```

Note - as of this writing (December 2017), pip version of visual-logging library is out of date, so you have to update default pip version with version from github - otherwise it doesn't log numpy arrays/OpenCV images correctly.

Make sure your PYTHONPATH environmental variable contains current directory, so Python can find modules you are trying to run. You can set PYTHONPATH by adding following to your `.bash_profile` (OSX) or `.bash_rc` (LINUX) scripts:  
`export PYTHONPATH="."`
