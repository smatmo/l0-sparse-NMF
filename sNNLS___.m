function H = sNNLS___(X,W,L)
%
% find an approximate solution to the problem,
%
% minimize ||X - W * H||_F
% s.t. W(:) >= 0, H(:) >= 0
%      sum(H(:,k) > 0) <= L, for all k,
%
% using the sparse nonnegative least squares algorithm (sNNLS) described
% in [1].
%
% References:
%
% [1] Peharz and Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% [2] Lawson and Hanson, "Solving Least Squares Problems", Prentice-Hall, 1974.
%
% [3] M. H. Van Benthem and M. R. Keenan, "Fast algorithm for the solution of
% large-scale non-negativity-constrained least squares problems", Journal
% of Chemometrics, 2004; 18: 441-450.
%
% Robert Peharz, 2011
%

H = sparseNNLS(X,W,[],[],L,L);
