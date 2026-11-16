"""A local, file-based Mongolian number plate monitor."""
import os

# This desktop app handles one inference at a time. BLAS defaults otherwise
# create many threads before torch.set_num_threads(), wasting RAM on small PCs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
__version__ = "2.8.1"
