function H = NMP(X,W,G,L,numIterations)
% H = NMP(X,W,G,L,numIterations)
%
% Nonnegative Matching Pursuit.
% Minimizes approximately sum(sum((X - W*H).^2)) w.r.t. H, s.t. all(H(:) >= 0)
% and all(sum(H > 0,1) <= L). This means the columns of X are coded using a
% nonnegative linear combination of maximal L columns out of W.
%
% Input:
% X: DxN data matrix
% W: DxK dictionary matrix
% G: Gram Matrix W'*W (when G=[], G is calculated by the function)
% L: Sparseness Factor (maximal number of nonzero entries in each column of H)
% numIterations: number of NMF iterations for nonnegative least squares (default 10)
%
% Output:
% H (sparse): KxN nonnegative sparse coding matrix
%
%
% see "Sparse Nonnegative Matrix Factorization Using L0-constraints",
% R. Peharz, M. Stark, F. Pernkopf, MLSP 2010.
%
% March 2010, by Robert Peharz
%

[D,N]=size(X);
[D,K]=size(W);
H = sparse(K,N);

if nargin < 5
    numIterations = 10;
end

if isempty(G)
    G = W'*W;
end

% code each column of X
for k = 1:N
    Cidx = [];
    C = [];
    
    % residual is first the whole data vector
    residual = X(:,k);
    projections0 = (X(:,k)' * W)';
    
    % collect L atoms
    for l = 1:L
        projections = residual' * W;
        [maxVal,idx] = max(projections);
        
        if maxVal < 0
            break;
        end
        
        % insert atom
        if ~any(Cidx == idx)
            Cidx = [Cidx; idx];
            C = [C; maxVal];
        end
        
        % this approximates nonnegative least squares (NMF for H)
        for n = 1:numIterations
            C = C .* projections0(Cidx) ./ (G(Cidx,Cidx) * C + 1e-9);
        end
        
        % calculate new residual
        residual = X(:,k) - W(:,Cidx) * C;
    end
    
    H(Cidx, k) = C;
end


