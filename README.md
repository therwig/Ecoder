-# Ecoder
ECON-T autoencoder model

## Setup 

### On VM `klijnsma-gpu3`

Get data and untar
```
mkdir data; cd data
wget https://www.dropbox.com/s/502o1h5y0ukkasf/ecoder.tar.gz 
tar -xvzf ecoder.tar.gz
mv uscms/home/kkwok/eos/ecoder/* .
```

Setup environment using miniconda3
```
source install_miniconda3.sh #if your first time
source setup.sh #also if your first time
conda activate ecoder-env
pip install keras tensorflow numba numpy pandas matplotlib tensorflow_model_optimization pillow ot
```
Setup qkeras (h/t Thea!):
```
git clone https://github.com/google/qkeras.git
cd qkeras
python setup.py build
python setup.py install --user
cd ..
```
### Setup on LPC 
If you are working on the LPC cluster working node, use the following scripts to setup the environment
```
source LPC_envSetup.sh      ##do this for the first time
source lpc_env.sh           ##do this everytime
```

## Juypter notebook demos
Following files illustrates prototypes of different autoencoder architectures

`auto.ipynb` - 1D deep NN autoencoder demo

`auto_CNN.ipynb` - 2D CNN autoencoder demo

`Auto_qCNN.ipynb` - 2D quantized CNN autoencoder, trained  with qKeras demo

qkeras instructions: https://github.com/google/qkeras

## Training scripts 
Scripts to explore hyperparameters choices:

`models.py`   - constructs and compile simple model architectures

`denseCNN.py` - model class for constructing conv2D-dense architectures

`train.py`    - train(or load weights) and evaluate models

## Example usage:

```
## edit parameters setting inside train.py
## train with 1 epoch to make sure model parameters are OK, output to a trainning folder
python train.py -i datadir/2elink/ -o output_folder --epoch 100 --quantize # last flag is for qkeras
## train the weights with max 150 epoch 
python train.py -i datadir/2elink/ -o output_folder --epoch 100 --quantize # last flag is for qkeras

## After producing a `.hdf5` file from trainning, you can re-run the model skipping the trainning phase.
## Do so by simply setting the model parameter 'ws' to `modelname.hdf5`
```
