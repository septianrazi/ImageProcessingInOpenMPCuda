# Image Processing in Open MP and Cuda


## Description

Implementation of various anaglyph and image processing techniques unto an image using OpenMp and Cuda

Can be accessed here: [https://septianrazi.github.io/AnaglyphVideoProcessingInThreeJS/](https://septianrazi.github.io/AnaglyphVideoProcessingInThreeJS/)

### Image Processing Techniques
We included the following techniques for our code

- Anaglyphs
- Gaussian Blur
- Covariance Denoising

Most of the code to accomplish this was done using fragment shader in the index.html file provided.


##### Exercise 1:
- [ex1_anaglyphy_gaussian.cpp](ex1_anaglyph_gaussian.cpp) for OpenMP implementation
- [ex1_anaglyph_gaussian.cu](ex1_anaglyph_gaussian.cu) for CUDA implementation

##### Exercise 2:
- [ex2_shared_memory](ex2_shared_memory.cu) for CUDA implementation using Shared Memory

##### Exercise 3:


## Prerequisites
- g++
- CUDA 11.6 nvcc
- OpenCV
- OpenMP


## Installation

Each module has a unique run command that can be executed in the command line. We recommend doing so on an Ubuntu Machine

1. [ex1_anaglyphy_gaussian.cpp](ex1_anaglyph_gaussian.cpp)

```
g++ ex1_anaglyph_gaussian -fopenmp `pkg-config opencv4 --cflags` -c 
g++ ex1_anaglyph_gaussian.o -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o ex1_anaglyph_gaussian
./ex1_anaglyph_gaussian test_stereo.jpg
```

2. [ex1_anaglyph_gaussian.cu](ex1_anaglyph_gaussian.cu) for CUDA implementation

```
/usr/local/cuda-11.6/bin/nvcc ex1_anaglyph_gaussian.cu `pkg-config opencv4 --cflags --libs` -o ex1
./ex1 test_stereo.jpg
```

3. [ex2_shared_memory](ex2_shared_memory.cu) for CUDA implementation using Shared Memory

```
/usr/local/cuda-11.6/bin/nvcc ex2_shared_memory.cu `pkg-config opencv4 --cflags --libs` -o ex2
./ex2 test_stereo.jpg
```

## Usage

Each of the image processing modules have the same command line functions. Adding -h or --help to any of the executables will show you the complete list.

```
Usage: program imagePath [-g gaussianKernel gaussianSigma] [-a anaglyphValue] [-d denoisingNbhoodSize denoisingFactorRatio]
Options:
  -g, --gaussian     Enable Gaussian filter with specified kernel and sigma
                      kernel: size of the Gaussian kernel (int, default: 4)
                      sigma: standard deviation of the Gaussian distribution (double, default: 2.0)
  -a, --anaglyph     Enable Anaglyph with specified value
                      anaglyphValue: value for the Anaglyph effect (int, default: 1)
                        1: True Anaglyph
                        2: Grey Anaglyph
                        3: Color Anaglyph
                        4: Half-Color Anaglyph
                        5: Optimised Anaglyph
  -d, --denoising    Enable Denoising with specified neighborhood size and factor ratio
                      neighbourhood size: size of the neighborhood for denoising (int, default: 8)
                      factor ratio: factor ratio for denoising (double, default: 60)
  -h, --help         Display this help message and exit
```

You can run these commands without the numerical arguments, and it will take the default values for these

```
program imagePath -a -g
```

You can also simply run the program without any image processing techniques to return the same image.


## License

This project is licensed under the [MIT License](LICENSE).

