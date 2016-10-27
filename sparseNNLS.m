function H = sparseNNLS(X,W,WtW,WtX,L,M,initH)
% 
% H = sparseNNLS(X,W,WtW,WtX,L,M,initH)
%
% Performs nonnegative least squares with multiple rhs [2], until maximal
% M basis vectors (columns of W) are collected, i.e. until we have maximal M 
% nonnegative coefficients in each column of H. Then iterativly discard 
% smallest coefficient and perform the NNLS "correction loop" until maximal 
% L nonzero coefficients remain in each column of H.
% This function unifies several methods, depending on parameters L and M:
%
% 1) for L = M = size(W,2), this function returns the nonnegative least 
% squares solution for multiple rhs, i.e. it solves w.r.t. H
%
%   minimize  ||X - W * H||_F
%   s.t.      W(:) >= 0
%             H(:) >= 0
%
% This is the algorithm described in [2].
%          
% 2) for L = M < size(W,2), this function returns an approximate solution
% for the nonnegative sparse coding problem, i.e.
%
%   minimize  ||X - W * H||_F
%   s.t.      W(:) >= 0
%             H(:) >= 0
%             sum(H(:,k) > 0) <= L   for all k
%
% This is the sparse NNLS (sNNLS) algorithm described in [3].
%
% 3) for L < M = size(W,2), this function returns an approximate solution
% for the nonnegative sparse coding problem, i.e.
%
%   minimize  ||X - W * H||_F
%   s.t.      W(:) >= 0
%             H(:) >= 0
%             sum(H(:,k) > 0) <= L   for all k
%
% This is the reverse sparse NNLS (rsNNLS) algorithm described in [3].
%
% For L < M < size(W,2), we get a trade-off solution between sNNLS and rsNNLS.
%
% 
% input: 
%        X:     D x N data matrix (should be nonnegative).
%        W:     D x K dictionary matrix (should be negative) .
%        WtW:   W'*W (if [] is passed, this will be calculated).
%        WtX:   W'*X (if [] is passed, this will be calculated).
%        L:     maximal allowed number of nonzeros per column of H.
%        M:     "forward" NNLS will break after M collected nonzeros
%        initH: initial H, for "warmstart"
%
% output:
%        H:     coding matrix, will be nonnegative and contain maximal
%               L nonzeros per column
%
% References:
%
% [1] Lawson and Hanson, "Solving Least Squares Problems", Prentice-Hall, 1974.
%
% [2] M. H. Van Benthem and M. R. Keenan, "Fast algorithm for the solution of
% large-scale non-negativity-constrained least squares problems", Journal
% of Chemometrics, 2004; 18: 441-450.
%
% [3] R. Peharz and F. Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% Robert Peharz, 2011
%

tol = 10*eps*norm(W,1)*length(W);

[~,K] = size(W);
[D,N] = size(X);

maxIter = 5*K;
iterCount = 0;

tmpVar = zeros(K,N);
tmpAlpha = inf(K,N);
G = zeros(K,N);

if nargin < 3 || isempty(WtW)
    WtW = W'*W;
end
if nargin < 4 || isempty(WtX)
    WtX = W'*X;
end
if nargin < 5 || isempty(L)
    L = K;
end
if nargin < 6 || isempty(M)
    M = L;
end

M = min([M,K,D]);

% Initialize in-active set of active columns to all
if nargin < 7 || size(initH,1) ~= K || size(initH,2) ~= N || any(initH(:) < 0)
    P = false(K,N);
    H = zeros(K,N);
else
    P = initH > 0;
    H = initH;
    
    G = combinatorialLS(WtW, WtX, P);
    
    negSet = find(any(G <= tol & P))';
    
    %%% correction loop
    while ~isempty(negSet)
        [rowIdx, colIdx] = find(P(:,negSet) & G(:,negSet) < tol);
        globCol = negSet(colIdx);
        idx = sub2ind([K,N], rowIdx, globCol);
        
        tmpAlpha(idx) = H(idx) ./ (H(idx) - G(idx));
        alpha = min(tmpAlpha(:,negSet));
        tmpAlpha(idx) = Inf;
        
        H(:,negSet) = H(:,negSet) + repmat(alpha,K,1) .* (G(:,negSet) - H(:,negSet));
        
        % Reset Z and P given intermediate values of H
        P(:,negSet) = abs(H(:,negSet)) >= tol & P(:,negSet);
        
        % Re-solve for G
        G(:,negSet) = combinatorialLS(WtW, WtX(:,negSet), P(:,negSet));
        
        negSet = find(any(G <= tol & P))';
    end
    H = G;
end

grad = WtX - WtW*H;
inIdx = find(any(grad > tol & ~P));

%%% -------------------------------------------------------------------
%%% Forward stage: active set NNLS, break when M nonzeros are 
%%% collected (sNNLS)
%%% -------------------------------------------------------------------

%%% outer loop: select next variable for active set
while ~isempty(inIdx)
    tmpVar(P) = -Inf;
    tmpVar(~P) = grad(~P);
    
    % Find variable with largest Lagrange multiplier
    [~,maxIdx] = max(tmpVar(:,inIdx));
    
    % Move variables zero set to positive set
    idx = sub2ind([K,N], maxIdx, inIdx);
    P(idx) = true;
    
    % Compute intermediate solution using only variables in positive set
    G(:,inIdx) = combinatorialLS(WtW, WtX(:,inIdx), P(:,inIdx));
    
    negSet = any(G(:,inIdx) <= tol & P(:,inIdx));
    negSet = inIdx(negSet)';
    
    %%% correction loop
    while ~isempty(negSet)
        iterCount = iterCount + 1;
        [rowIdx, colIdx] = find(P(:,negSet) & G(:,negSet) < tol);
        globCol = negSet(colIdx);
        idx = sub2ind([K,N], rowIdx, globCol);
        
        tmpAlpha(idx) = H(idx) ./ (H(idx) - G(idx));
        alpha = min(tmpAlpha(:,negSet));
        tmpAlpha(idx) = Inf;
        
        H(:,negSet) = H(:,negSet) + repmat(alpha,K,1) .* (G(:,negSet) - H(:,negSet));
        
        % Reset Z and P given intermediate values of H
        P(:,negSet) = abs(H(:,negSet)) >= tol & P(:,negSet);
        
        % Re-solve for G
        G(:,negSet) = combinatorialLS(WtW, WtX(:,negSet), P(:,negSet));
        
        negSet = any(G(:,inIdx) <= tol & P(:,inIdx));
        negSet = inIdx(negSet)';
    end
    
    H = G;
    grad = WtX - WtW*H;
    inIdx = find(any(grad > tol & ~P) & sum(P) < M);
    
    if iterCount > maxIter
        % fprintf('MAX ITERCOUNT\n')
        break
    end
end

%%% -----------------------------------------------------------------------
%%% Reverse stage: remove coefficients until maximal L nonzeros 
%%% remain (rsNNLS)
%%% -----------------------------------------------------------------------

inIdx = find(sum(P) > L);
while ~isempty(inIdx)
    tmpVar = H;
    tmpVar(~P) = Inf;
    [~,minIdx] = min(tmpVar(:,inIdx));
    
    idx = sub2ind([K,N], minIdx, inIdx);
    
    P(idx) = false;
    
    G(:,inIdx) = combinatorialLS(WtW, WtX(:,inIdx), P(:,inIdx));
    negSet = any(G(:,inIdx) <= tol & P(:,inIdx));
    negSet = inIdx(negSet)';
    
    %%% correction loop
    while ~isempty(negSet)
        [rowIdx, colIdx] = find(P(:,negSet) & G(:,negSet) < tol);
        globCol = negSet(colIdx);
        idx = sub2ind([K,N], rowIdx, globCol);
        
        tmpAlpha(idx) = H(idx) ./ (H(idx) - G(idx));
        alpha = min(tmpAlpha(:,negSet));
        tmpAlpha(idx) = Inf;
        
        H(:,negSet) = H(:,negSet) + repmat(alpha,K,1) .* (G(:,negSet) - H(:,negSet));
        
        % Reset Z and P given intermediate values of H
        P(:,negSet) = abs(H(:,negSet)) >= tol & P(:,negSet);
        
        % Re-solve for G
        G(:,negSet) = combinatorialLS(WtW, WtX(:,negSet), P(:,negSet));
        
        negSet = any(G(:,inIdx) <= tol & P(:,inIdx));
        negSet = inIdx(negSet)';
    end
    
    H = G;
    inIdx = find(sum(P) > L);
end

