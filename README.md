# GPU Computing with CUDA

This repository contains CUDA code samples.

**🎥 Course Reference:**  

[GPU Computing course (Spring 2021) by Izzat El Hajj](https://youtu.be/c8dehGOB8mQ?si=hN0lNLLc1ef0VAq4)

---

## Requirements

- **CUDA-capable GPU**  
- **nvcc compiler**

> No GPU? You can run the samples on **Google Colab**:  
> [Practice CUDA without a GPU](https://www.reddit.com/r/MachineLearning/comments/151n4te/d_practice_cuda_without_an_actual_nvidia_gpu/)

---

## Highlights

- Parallel implementation of **RGB to grayscale conversion**  
- **Matrix multiplication** with tiling and optimization  
- **Image convolution and blurring** 
- **Stencil computations** (3D)  
- **Performance profiling** with NVIDIA tools
- **Parallel patterns**

**💡 Article:**  
[Mastering Matrix Multiplication in CUDA](https://marshall5.medium.com/mastering-matrix-multiplication-in-cuda-13275162c1cc)

---

## Structure

```text
parallel-patterns/      # Parallel pattern examples
*.cu, *.cpp             # CUDA and C++ programs
*.png, *.jpg            # Example images
run.ipynb               # Notebook for quick testing
*.prof, timeline.prof   # Profiling outputs
