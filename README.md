# Globall Histogram Equalization

This repository contains Python implementation of global histogram equalization for AI6121 - Computer Vision course assignment (NTU MSc in AI programme). I have also written a Medium article explaining the concepts. 

[Introduction to Histogram Equalization for Digital Image Enhancement](https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43)

## Usage
The histogram equalizer is implemented in histogram equalizer.py. The script to process the test images is CV Assignment1.py. The test images should be placed in jpg/ folder. An output folder jpg/equalized/ must be created, where the equalized images will be written to. To use the equalizer, the following command can be coded:

```
from histogram equalizer import histogram equalizer

#to use the histogram equalizer
eq_img_array , transform map , cdf , pdf = histogram equalizer(img array)
```

### Input:
- ```img array```: 2D NumPy array of values to be equalized, in the range of 0 to 255 (uint8). If the array is multi-channel (eg. RGB image array), each channel will be equalized independently.
### Output:
- ```eq img array```: NumPy array of the equalized values with the same shape as the input
array
- ```transform map``` : the mapping array to transform the intensity values
- ```cdf``` : cumulative density function of the input array
- ```pdf``` : normalized value distribution of the input array

## Dependencies: 
The scripts depends on the following Python packages: 
- NumPy
- PIL
- skimage
- matplotlib
- pandas

On conda environment, the dependencies can be installed via the provided requirements.txt
file with the following command:

```
pip install −r requirements.txt
```

## Demo

A simple demo is provided in Jupyter notebook, found in the folder /demo_notebook/
