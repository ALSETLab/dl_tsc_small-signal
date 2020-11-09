Time Series-Based Small-Signal Stability Assessment using Deep Learning
===========================================

### Installation

Run:

```
python pip -r install requirements.txt
```

This code runs with Tensorflow 2.0. It was tested with Ubuntu 18.04 LTS. This updated Tensorflow release is automatically capable of GPU computation. To install the CUDA dependencies via `conda`, do :`conda install cudatoolkit=10.1 cudnn` *before* installing the package requirements of this repository.

### Execution

```
python main.py <model_name> <data_set> --validation --folds --verbose
```

### Data Preprocessing

The data preprocessing scripts allow to:

- Reduce the length of the time series data (`--n_points`)
- Reduce the number of instances in the training set (`--n_reduction`)

For example, to create a dataset using the noisy data with 400 points and reducing by 2 the number of training instances, do:

```
python _preprocessing.py IEEE9_noisy --n_points 400 --n_reduction 2

```

This will create a new folder inside `train_data` called `IEEE9_noisy_2red_500` (note how the reduction factor and the number of points are included in the name of the folder). To train a NN using this dataset, run:

```
python main.py fcn IEEE9_noisy_2red_500
```

### Data Loading

The supplementary script `_load_data.py` loads the pickle data for a given dataset in case the NumPy arrays cannot be loaded.

```
python _load_data.py IEEE9_noisy_2red_500
```
