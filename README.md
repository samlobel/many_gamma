# Many Gamma

# Install instructions
```
python -m venv venv
source ./venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


That was wrong because of cudnn stuff. Trying cenv instead
```
conda create --prefix ./cenv python=3
conda activate ./cenv
<!-- conda install -c conda-forge cudnn -->
# conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
<!-- conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -->
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install --upgrade pip setuptools wheel
pip install -r requirements.pip
<!-- pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -->
```

Problems abound. Following tf instructions: https://www.tensorflow.org/install/pip
Seems like the problem is with jax maybe, which needs 8.2 or newer. So what? I guess rebuild with that.
```
conda create --prefix ./cenv python=3
conda activate ./cenv
# conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c conda-forge cudnn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install --upgrade pip setuptools wheel
pip install -r requirements.pip
