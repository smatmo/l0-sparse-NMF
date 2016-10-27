%
% simple example application of NMFL0-H
%

clear all

%%% data dimensionality
D = 100;

%%% number data samples
N = 1000;

%%% dictionary size
K = 400

%%% maximal allowed number of nonzeros per column of H
L = 10;

%%% generate dictionary
W = createDictionaryRand(D,K);

%%% make "true" coding matrix
Htrue = zeros(K,N);
for n = 1:N
    rp = randperm(K);
    Htrue(rp(1:L),n) = 10*abs(randn(L,1));
end

%%% make synthetic data
X = W * Htrue;
frobX = norm(X,'fro');

%%% set NMFL0_H parameters
options.K = K;
options.L = L;
options.numIter = 25;

%%% select sparse coder
options.sparseCoder = @rsNNLS___;
% options.sparseCoder = @sNNLS___;

%%% select dictionary update method
options.updateType = 'ANLS_FC';
% options.updateType = 'MU';
% options.updateType = 'KSVD';
% options.updateType = 'ANLS_PG';
% options.NNLS_PG_tolerance = 1e-3;
% options.NNLS_PG_maxIter = 100;

%%% number of dictionary updates
options.numUpdateIter = 10;


[W,H,INFO] = NMFL0_H(X, options);


