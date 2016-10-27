function W = createDictionaryRand(D,K)
%
% W = createDictionaryRand(D,K)
%
% Creates a random toy dictionary used in
% R. Peharz and F. Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% Robert Peharz, 2011
%

W = abs(randn(D,K));
W = W * diag(1./sqrt(sum(W.^2)));