function [W,H,INFO] = NMFL0_H(X, options)
%
% [W,H,INFO] = NMFL0_H(X, options)
%
% run NMFL0_H described in 
% R. Peharz and F. Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% The algorithm returns an approximate solution for 
%
%   minimize  ||X - W * H||_F
%   s.t.      W(:) >= 0
%             H(:) >= 0
%             sum(H(:,k) > 0) <= L   for all k
%
% w.r.t. W and H
%
%
% input:
%
%  X nonnegative data matrix 
%  options: structure of parameters:
%    K:                 number of columns in dictionary matrix W
%    L:                 maximal number of nonzeros in each column of H
%    numIter:           number of (outer) iterations
%    sparseCoder:       function handle to sparse coder;  sparse coder function 
%                       must have the form  H = foo(X,W,L,[params]), and
%                       return an approximation to the problem
%      
%                         minimize  ||X - W * H||_F
%                         s.t.      W(:) >= 0
%                         H(:) >= 0
%                         sum(H(:,k) > 0) <= L   for all k
%
%                       w.r.t. H.
%    sparseCoderParams: params to be passed to the sparse coder. If not
%                       defined, the sparse coder will be called without params
%    updateType:        update type for dictionary W updates.
%                       - 'MU'multiplicative updates according to Lee and Seung, 
%                        "Algorithms for nonnegative matrix factorization", 2001.
%                       - 'ANLS_FC': alternating nonnegative least squares, using
%                        fast combinatorial approach for NNLS, by  
%                        M . H. Van Benthem and M. R. Keenan, "Fast algorithm 
%                        for the solution of large-scale 
%                        non-negativity-constrained least squares problems", 
%                        Journal of Chemometrics, 2004
%                       - 'ANLS_PG': alternating nonnegative least squares, using
%                        projected gradient approach by Chih-Jen Lin, 
%                        "Projected Gradient Methods for Nonnegative Matrix
%                        Factorization", Neural Computation, 2007.
%                        For this update method, options must also contain the 
%                        fields 
%                        NNLS_PG_tolerance 
%                        NNLS_PG_maxIter 
%                        (see NNLS_PG.m).
%                       - 'NNKSVD': dictionary updates from Aharon and Bruckstein,
%                        "K-SVD and its non-negative variant for dictionary
%                        design", SPIE, 2005.
%    numUpdateIter:     number of update (innner) iterations; only relevant if
%                       timeBudgetUpdate is empty
%    timeBudgetUpdate:  if not empty, this contains a vector of length
%                       numIter, containing the time budget (in seconds) 
%                       to be used for the dictionary update in each outer
%                       iteration.
%    VERBOSITY:         verbose mode if not 0; default 1.
%    W:                 initial dictionary to be used ([] if none). 
%    initType:          dictionary initialisation if field W is empty; 
%                       'rand' for random numbers or 'samples' for data 
%                       samples (default)
%    
%
% output:
%
%  W:       dictionary matrix
%  H:       coding matrix
%  INFO:    structure of some info
%     E_SC:   error after sparse coder in each iteration (||X - W*H||_F)
%     E:      error in each iteration ||X - W*H||_F)
%     SCtime: time needed by sparse coder
%     UDtime: time needed by dictionary updates
%
% Robert Peharz, 2011
%

if any(X(:)<0),                                  error('X contains negative values.'); end
if ~isfield(options,'K'),                        error('options must contain parameter K (number of basis vectors).'); end
if ~isfield(options,'L'),                        error('options must contain parameter L (maximal number of basis vectors per data sample).'); end
if ~isfield(options,'numIter'),                  error('options must contain parameter numIter.'); end
if ~isfield(options,'sparseCoder'),              error('options must contain parameter sparseCoder (function_handle).'); end
if ~isfield(options,'updateType'),               error('options must contain parameter updateType (string).'); end
if ~isfield(options,'numUpdateIter') && ~isfield(options,'timeBudgetUpdate')
                                                 error('options must contain either parameter numUpdateIter or timeBudgetUpdate.');
end

switch options.updateType
    case 'MU'
        
    case 'ANLS_FC'
        
    case 'ANLS_PG'
        if ~isfield(options,'NNLS_PG_tolerance'), error('options must contain parameter NNLS_PG_tolerance for updateType == ''ANLS_PG''.'); end
        if ~isfield(options,'NNLS_PG_maxIter'),   error('options must contain parameter NNLS_PG_maxIter for updateType == ''ANLS_PG''.'); end
        NNLS_PG_tolerance = options.NNLS_PG_tolerance;
        NNLS_PG_maxIter = options.NNLS_PG_maxIter;
    case 'NNKSVD'
        
    otherwise
        error('unknown update method.')
end

sparseCoder = options.sparseCoder;

if isfield(options,'timeBudgetUpdate')
    timeBudgetUpdate = options.timeBudgetUpdate;
    options.numUpdateIter = [];
else
    timeBudgetUpdate = [];
end

if isfield(options,'sparseCoderParams')
    sparseCoderParams = options.sparseCoderParams;
else
    sparseCoderParams = [];
end

if isfield(options,'verbosity')
    VERBOSITY = options.verbosity;
else
    VERBOSITY = 1;
end

[D,N] = size(X);
K =             options.K;
L =             options.L;
numIter =       options.numIter;
updateType =    options.updateType;
numUpdateIter = options.numUpdateIter;

if nargout > 2
    E = zeros(numIter,1);
    E_SC = zeros(numIter,1);
    SCtime = zeros(numIter,1);
    UDtime = zeros(numIter,1);
    INFO = [];
end

if K >= N
    H = diag(sqrt(sum(X.^2)));
    H = [H; zeros(K-N,N)];
    
    W = X;
    W = W .* diag(1./sqrt(sum(W.^2)));
    W = [W, ones(D,K-N) / D];
    
    warning('too little data, trivial solution is returned.')
    return
end

if isfield(options,'initType') && strcmp(options.initType, 'rand')
    initType = 'rand';
else
    initType = 'samples';
end

if isfield(options,'W')
    if any(options.W(:)<0), error('options.W contains negative values.'); end
    W = options.W;
else
    if strcmp(initType, 'rand')
        W = rand(D,K);
    else
        rp = randperm(N);
        W = X(:,rp(1:K));
    end
end
W = W * diag(1./sqrt(sum(W.^2)));

%%%-----------------------------------------------------------------%%%

for iter = 1:numIter
    
    if VERBOSITY
        fprintf('Iteration: %d   ',iter);
    end
    
    %%% Sparse Coder Stage
    tic
    if isempty(sparseCoderParams)
        H = sparseCoder(X,W,L);
    else
        H = sparseCoder(X,W,L,sparseCoderParams);
    end
    elapsedT = toc;
    
    if nargout > 2
        E_SC(iter) = norm(X-W*H,'fro');
        SCtime(iter) = elapsedT;
    end
    
    if VERBOSITY
        fprintf('SC-Error: %d   ',norm(X-W*H,'fro')/norm(X,'fro'));
    end
    
    unusedIdx = all(H == 0,2);
    numUnused = sum(unusedIdx);
    W = W(:,~unusedIdx);
    H = H(~unusedIdx,:);
    
    %%% Update Stage
    switch updateType
        case 'MU'
            if isempty(timeBudgetUpdate)
                tic
                for k = 1:numUpdateIter
                    W = W .* ((X*H') ./ (W*H*H' + 1e-12));
                    if k < numUpdateIter
                        H = H .* ((W'*X) ./ (W'*W*H + 1e-12));
                    end
                end
                elapsedT = toc;
            else
                tic
                while toc < timeBudgetUpdate(iter)
                    W = W .* ((X*H') ./ (W*H*H' + 1e-12));
                    H = H .* ((W'*X) ./ (W'*W*H + 1e-12));
                end
                elapsedT = toc;
            end
            
            
        case 'ANLS_FC'
            if isempty(timeBudgetUpdate)
                tic
                mask = H > 0;
                for k = 1:numUpdateIter
                    %fprintf('update W\n')
                    Wt = sparseNNLS(X',H',[],[],K,K,W');
                    W = Wt';
                    if k < numUpdateIter
                        %fprintf('update H\n')
                        H = maskNNLS(X,W,[],[],mask,H);
                    end
                end
                elapsedT = toc;
            else
                tic
                mask = H > 0;
                while toc < timeBudgetUpdate(iter)
                    fprintf('.')
                    Wt = sparseNNLS(X',H',[],[],K,K,W');
                    W = Wt';
                    
                    if toc >= timeBudgetUpdate(iter)
                        break
                    end
                    
                    H = maskNNLS(X,W,[],[],mask,H);
                end
                fprintf('\n')
                elapsedT = toc;
            end
            
        case 'ANLS_PG'
            %%% no time budget variant here
            if isempty(numUpdateIter)
                error('time budget variant not implemented for ANLS via projected gradient.\n');
            end
            tic
            mask = H > 0;
            for k = 1:numUpdateIter
                Wt = NNLS_PG(X',H',W',NNLS_PG_tolerance,NNLS_PG_maxIter);
                W = Wt';
                if k < numUpdateIter
                    H = NNLS_PG_mask(X,W,H,NNLS_PG_tolerance,NNLS_PG_maxIter,mask);
                end
            end
            elapsedT = toc;
            
            
        case 'NNKSVD'
            K = size(W,2);
            if isempty(timeBudgetUpdate)
                tic
                for k = 1:K
                    idx = H(k,:) > 0;
                    Etild = X(:,idx) - W(:,1:k-1) * H(1:k-1,idx) - W(:,k+1:end) * H(k+1:end,idx);
                    
                    [U,S,V] = svds(Etild,1);
                    V = V'*S;
                    
                    pu = max(U,0);
                    pv = max(V,0);
                    pu2 = max(-U,0);
                    pv2 = max(-V,0);
                    
                    if norm(Etild - pu2 * pv2,'fro') < norm(Etild - pu * pv,'fro')
                        pu = pu2;
                        pv = pv2;
                    end
                    
                    for l=1:numUpdateIter
                        pu = Etild * pv'  / (pv*pv');
                        pv = pu' * Etild / (pu'*pu);
                        pu = max(pu,0);
                        pv = max(pv,0);
                    end
                    
                    H(k,idx) = pv * norm(pu);
                    W(:,k) = pu / norm(pu);
                end
                elapsedT = toc;
            else
                elapsedT = 0;
                rp = randperm(K);
                for kcount = 1:K
                    tic
                    
                    k = rp(kcount);
                    
                    idx = H(k,:) > 0;
                    Etild = X(:,idx) - W(:,1:k-1) * H(1:k-1,idx) - W(:,k+1:end) * H(k+1:end,idx);
                    
                    [U,S,V] = svds(Etild,1);
                    V = V'*S;
                    
                    pu = max(U,0);
                    pv = max(V,0);
                    pu2 = max(-U,0);
                    pv2 = max(-V,0);
                    
                    if norm(Etild - pu2 * pv2,'fro') < norm(Etild - pu * pv,'fro')
                        pu = pu2;
                        pv = pv2;
                    end
                    
                    while toc < timeBudgetUpdate(iter) / K
                        pu = Etild * pv'  / max((pv*pv'),1e-12);
                        pv = pu' * Etild / max((pu'*pu),1e-12);
                        pu = max(pu,0);
                        pv = max(pv,0);
                    end
                    
                    H(k,idx) = pv * norm(pu);
                    W(:,k) = pu / norm(pu);
                    elapsedT = elapsedT + toc;
                end
            end
    end
    
    if nargout > 2
        E(iter) = norm(X-W*H,'fro');
        UDtime(iter) = elapsedT;
    end
    
    if VERBOSITY
        fprintf('Update-Error: %d\n',norm(X-W*H,'fro')/norm(X,'fro'));
    end
    
    %%% Reinitialize
    if strcmp(initType, 'rand')
        W = [W,rand(D,numUnused)];
    else
        rp = randperm(N);
        W = [W,X(:,rp(1:numUnused))];
    end
    
    W = W * diag(1./sqrt(sum(W.^2)));
end

if nargout > 2
    INFO.E = E;
    INFO.E_SC = E_SC;
    INFO.SCtime = SCtime;
    INFO.UDtime = UDtime;
end


