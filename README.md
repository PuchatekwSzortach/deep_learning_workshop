# `deep_learning_workshop`

A set of files with code I use when teaching deep learning

Interesting files:  
- `code/net.py` - contains implementation of a simple neural network class  
- `code/mnist.py` - trains a simple network on mnist dataset, outputs result to console and `./data/mnist.html`  
- `code/simple_tensorflow.py` - code demonstrating use of placeholders, variables and sessions in tensorflow  
- `code/mnist_tensorflow.py` - trains a simple network built in tensorflow on mnist dataset, outputs result to console and `./data/mnist_tensorflow.html`

### Environment definition

Environment management is done using Anaconda. Please make sure to install Anaconda on your system first.
Once Anaconda is installed, you can use environment definition file for your operating system to install all project dependencies.

On OSX:
```bash
conda env create -f environment_osx.yml
source activate deep_learning_workshop_osx
```

On Linux:
```bash
conda env create -f environment_linux.yml
source activate deep_learning_workshop_linux
```

On Windows:
```bash
conda env create -f ./environment_windows.yml
conda activate deep_learning_workshop_windows
```

Should you want to create your own environment from scratch, here are the major dependencies:  
- python 3.5  
- opencv  
- tensorflow 1.3+  
- numpy  
- tqdm  
- visual-logging  

Make sure your `PYTHONPATH` environmental variable contains current directory, so Python can find modules you are trying to run. You can set PYTHONPATH by adding following to your `.bash_profile` (OSX) or `.bash_rc` (LINUX) scripts:  
`export PYTHONPATH="."`
