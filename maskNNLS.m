function H = maskNNLS(X,W,WtW,WtX,mask,initH)
%
% H = maskNNLS(X,W,WtW,WtX,mask,initH)
% 
% solve 
%
% minimize  ||X - W * H||_2
% s.t.      W(:) >= 0
%           H(:) >= 0
%           H(~mask) = 0
%
% This is used for the sparseness maintaining dictionary (coding matrix)
% updates, used in [3].
%
% input: 
%        X:     D x N data matrix (should be nonnegative).
%        W:     D x K dictionary matrix (should be negative) .
%        WtW:   W'*W (if [] is passed, this will be calculated).
%        WtX:   W'*X (if [] is passed, this will be calculated).
%        mask:  K x N logical matrix; H(~mask) will be constrained to be 0
%        initH: initial H, for "warmstart"
%
% output:
%        H:     coding matrix
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
[~,N] = size(X);

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

% Initialize
if nargin < 6 || size(initH,1) ~= K || size(initH,2) ~= N || any(initH(:) < 0)
    P = false(K,N);
    H = zeros(K,N);
else
    P = initH > 0 & mask;
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
inIdx = find(any(grad > tol & ~P & mask));

%%%-------------------------------------------------------------%%%

%%% outer loop: select next variable for active set
while ~isempty(inIdx)
    tmpVar(P | ~mask) = -Inf;
    tmpVar(~P & mask) = grad(~P & mask);
    
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
    inIdx = find(any(grad > tol & ~P & mask));
    
    if iterCount > maxIter
        %fprintf('MAX ITERCOUNT\n')
        break
    end
end


