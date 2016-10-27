function s = hoyerS(X)
%
% s = hoyerS(X)
% 
% Calculate the sparsness of the column vectors in X according to the sparsness measure 
% defined in
%
% "Non-negative Matrix Factorization with Sparseness Constraints", P. Hoyer,
% Journal of Machine Learning Research 5, 2004.
%

[D,N] = size(X);
s = zeros(1,N);
s = (sqrt(D) - sum(abs(X)) ./ sqrt(sum(X.^2))) / (sqrt(D)-1);
