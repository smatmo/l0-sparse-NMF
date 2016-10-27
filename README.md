# l0-sparse-NMF

This package reproduces results from 

Robert Peharz and Franz Pernkopf,
"Sparse Nonnegative Matrix Factorization with l0-constraints", 
Neurocomputing, vol. 80, pp. 38--46, 2012.

In particular, it provides algorithms for approximate non-negative matrix factorization with l0-sparseness constraints.

PLEASE NOTE THE ACCOMPANYING LICENSE FILE (modified BSD, 3-Clause). 
IF YOU USE THIS CODE FOR RESEARCH, PLEASE CITE THE PAPER ABOVE.


Overview:

NMFL0_H.m: implements approximate NMF with l0-sparseness constraints on the columns of H. See help text in m-file for further information.

NMFL0_W.m: implements approximate NMF with l0-sparseness constraints on the columns of W. See help text in m-file for further information.

sparseNNLS.m: implements several functions, such as nonnegative least squares (NNLS), sparse nonnegative least squares (sNNLS) and reverse sparse nonnegative least squares (rsNNLS). See help text in m-file for further information.

experiment_SparseCoder_SyntheticData.m: reproduces experiment in section 4.1
experiment_NMFL0_H_spectrogram.m: reproduces experiment in section 4.2
experiment_NMFL0_W_ORLFaces.m: reproduces experiment in section 4.3

example_*.m: shorter application examples
