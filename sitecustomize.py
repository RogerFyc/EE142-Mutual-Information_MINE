import os

# Avoid duplicated OpenMP runtime error on Windows when PyTorch,
# NumPy, scikit-learn, matplotlib, or torchvision load different OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")