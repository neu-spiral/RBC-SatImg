# Recursive Water and Land Mapping

This code has been implemented in Python 3.6.13, using the IDE Pycharm 2021.2.3. 

## Project Structure
The project is structured as follows.

* `README.md`

* `requirements.txt`

* `./benchmark/`

    * `./deepwatermap_main/` contains part of the deepwatermap algorithm open 
source code shared in [this GitHub repository](https://github.com/isikdogan/deepwatermap). An `__init__.py`
file has been added to this directory to treat it as a module.
    
    * `./watnet/` contains part of the WatNet algorithm open source code shared 
in [this GitHub repository](https://github.com/xinluo2018/WatNet). An `__init__.py`
file has been added to this directory to treat it as a module.

    * `benchmark.py` includes a copy of two functions from
`./deepwatermap_main/inference.py`. The `main` function contains one change.

* `./logs/` contains the `log` files generated at every code execution.

* `./tools/` contains useful functions.
    
    * `operations.py`

    * `path_operations.py`

    * `spectral_index.py`

* `configuration.py` contains the classes `Debug` and `Config`. Some configuration settings 
must be changed when executing the code with data that is different to the one provided by the 
authors.

* `main.py` contains the main logic and flow of the code.

* `bayesian_recursive.py` contains the core of the recursive bayesian algorithm for 
classification.

* `figures.py` includes functions to plot the generated results.

* `image_reader.py` contains the abstract class `ImageReader` and the class `ReadSentinel2` which
allows the user to read images from a dataset of Sentinel2 images, such as the one provided
by the authors of this code.

* `./trained_models/` contains `pickle` files with saved data from the 
training stage. If wanting to train the models from scratch, it should be indicated in 
the `Debug` class from `configuration.py`. Data has been stored in this file because the
training stage takes long.

* `training.py` contains functions that are linked to the training stage.

* `evaluation.py` contains functions that are linked to the evaluation stage.

## Installation

### Geospatial Data Abstraction Library (GDAL) Installation
Follow these instructions to [install GDAL for Python with pip on Windows](https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/)
or to [install GDAL for Python with Anaconda](https://opensourceoptions.com/blog/how-to-install-gdal-with-anaconda/). We recommend to create
a conda environment with Python 3.8 and run the command `conda install -c conda-forge gdal` in the Anaconda prompt.

### Other Packages
There are other packages besides GDAL that need to be installed. Required packages can be installed using the Python package installer `pip`:

<code>pip install -r requirements.txt</code>

### Dataset
Download our dataset *Sentinel-2 Images from Oroville Dam and Charles River* from [this Zenodo link](https://zenodo.org/record/6999172#.YzWyndjMI2x) and extract the `.zip` file. 
In `configuration.py` (class `Config`), change `path_sentinel_images` to the `sentinel_data` folder path. Images in this dataset are used for
training and evaluation. Details regarding the dataset can be found in the Zenodo link.

### Reproduction of Results
The results presented in our publication can be obtained by executing the `main_notebook.ipynb` (Jupyter Notebook) or the `main.py`file (Python script). We recommend to use Jupyter Notebook in a conda environment (see instructions [here](https://stackoverflow.com/questions/58068818/how-to-use-jupyter-notebooks-in-a-conda-environment)).

## Results
A log file is generated in the `path_log_files` path (defined in `configuration.py`, class `Config`) everytime the main scripts are executed. Log files contain information
regarding evenets in the code execution.

## References

The open source codes of the DeepWaterMap and WaterNet algorithms, used for benchmarking,
were provided by their respective authors.

* DeepWaterMap [(see the GitHub repository)](https://github.com/isikdogan/deepwatermap), by 
L. F. Isikdogan, A.C. Bovik and P. Passalacqua. This algorithm for water mapping is proposed
in the following publications:
    
    * [Seeing Through the Clouds With DeepWaterMap](https://ieeexplore.ieee.org/document/8913594)

    * [Surface Water Mapping by Deep Learning](https://ieeexplore.ieee.org/document/8013683)
    
* WatNet [(see the GitHub repository)](https://github.com/xinluo2018/WatNet), by
Xin Luo, Xiaohua Tong and Zhongwen Hu. This algorithm for water mapping is proposed in the publication
[An applicable and automatic method for earth surface water mapping based on multispectral images](https://www.sciencedirect.com/science/article/pii/S0303243421001793).

## Authors
* Bhavya Duvvuri, from The Beighley Lab (Sustainable Water Resources | Resilient Wet Infrastructure) at Northeastern
University, Boston (MA).
* Helena Calatrava, from the Signal Processing, Imaging, Reasoning and Learning (SPIRAL) Group also at 
Northeastern University, Boston (MA). 
