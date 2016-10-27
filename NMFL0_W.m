function [W,H,INFO] = NMFL0_W(X, options)
%
% [W,H,INFO] = NMFL0_W(X, options)
%
% run NMFL0_W described in 
% R. Peharz and F. Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% The algorithm returns an approximate solution for 
%
%   minimize  ||X - W * H||_2
%   s.t.      W(:) >= 0
%             H(:) >= 0
%             sum(W(:,k) > 0) <= L   for all k
%
% w.r.t. W and H
%
%
% input:   
%  X nonnegative data matrix 
%  options: structure of parameters:
%    K:                 number of columns in dictionary matrix W
%    L:                 maximal number of nonzeros in each column of W
%    numIter:           number of (outer) iterations
%    update type:       - 'MU'multiplicative updates according to Lee and Seung, 
%                       "Algorithms for nonnegative matrix factorization", 2001.
%                       - 'ANLS_FC': alternating nonnegative least squares, using
%                       fast combinatorial approach for NNLS, by  
%                       M . H. Van Benthem and M. R. Keenan, "Fast algorithm 
%                       for the solution of large-scale 
%                       non-negativity-constrained least squares problems", 
%                       Journal of Chemometrics, 2004
%                       - 'ANLS_PG': alternating nonnegative least squares, using
%                       projected gradient approach by Chih-Jen Lin, 
%                       "Projected Gradient Methods for Nonnegative Matrix
%                       Factorization", Neural Computation, 2007.
%                       For this update method, options must also contain the 
%                       fields NNLS_PG_tolerance and NNLS_PG_maxIter 
%                       (see NNLS_PG.m).
%    numUpdateIter:     number of update (innner) iterations.
%    VERBOSITY:         verbose mode if not 0; default 1.
%    H:                 initial H; if empty, random numbers are used for
%                       initialization.
%
% output:
%  W:       dictionary matrix
%  H:       coding matrix
%  INFO:    structure of some info
%     E:      error in each iteration ||X - W*H||_F)
%     UDtime: time needed by dictionary updates
%
% Robert Peharz, 2011
%

if any(X(:)<0),                                  error('X contains negative values.'); end
if ~isfield(options,'K'),                        error('options must contain parameter K (number of bases).'); end
if ~isfield(options,'L'),                        error('options must contain parameter L (maximal number of bases per signal).'); end
if ~isfield(options,'numIter'),                  error('options must contain parameter numIter.'); end
if ~isfield(options,'updateType'),               error('options must contain parameter updateType (string).'); end
if ~isfield(options,'numUpdateIter'),            error('options must contain parameter numUpdateIter.'); end

switch options.updateType
    case 'MU'        
        
    case 'ANLS_FC'
        
    case 'ANLS_PG'
        if ~isfield(options,'NNLS_PG_tolerance'), error('options must contain parameter NNLS_PG_tolerance for updateType == ''ANLS_PG''.'); end
        if ~isfield(options,'NNLS_PG_maxIter'),   error('options must contain parameter NNLS_PG_maxIter for updateType == ''ANLS_PG''.'); end
        NNLS_PG_tolerance = options.NNLS_PG_tolerance;
        NNLS_PG_maxIter = options.NNLS_PG_maxIter;
    otherwise
        error('unknown update method.')
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
    UDtime = zeros(numIter,1);
    INFO = [];
end

if isfield(options,'H')
    if any(options.H(:)<0), error('options.H contains negative values.'); end
    H = options.H;
else
    H = rand(K,N);
end

for iter=1:numIter
    if VERBOSITY
        fprintf('Iteration: %d   ',iter);
    end
    
    W = sparseNNLS(X',H',[],[],K,K);
    W = W';
       
    % Set K-L smallest values to zero for each atom
    for p=1:K
        [~, idx] = sort(W(:,p),'ascend');
        W(idx(1:D-L),p) = 0;
    end
    
    %%% Update Stage
    switch updateType
        case 'MU'
            tic
            for k=1:numUpdateIter
                H = H .* ((W'*X) ./ (W'*W*H + 1e-12));
                if k < numUpdateIter
                    W = W .* ((X*H') ./ (W*H*H' + 1e-12));
                end
            end
            elapsedT = toc;
            
        case 'ANLS_FC'
            tic
            mask = W' > 0;
            for k=1:numUpdateIter
                H = sparseNNLS(X,W,[],[],K,K);
                if k < numUpdateIter
                    W = maskNNLS(X',H',[],[],mask);
                    W = W';
                end
            end
            elapsedT = toc;
            
        case 'ANLS_PG'
            tic
            mask = W' > 0;
            for k=1:numUpdateIter
                H = NNLS_PG(X,W,H,NNLS_PG_tolerance,NNLS_PG_maxIter);                
                if k < numUpdateIter
                    W = NNLS_PG_mask(X',H',W',NNLS_PG_tolerance,NNLS_PG_maxIter,mask);
                    W = W';
                end
            end
            elapsedT = toc;
    end
    
    if nargout > 2
        E(iter) = norm(X-W*H,'fro');
        UDtime(iter) = elapsedT;
    end
    
    if VERBOSITY
        fprintf('Update-Error: %d\n',norm(X-W*H,'fro')/norm(X,'fro'));
    end
end

normW = sqrt(sum(W.^2));
H = diag(normW) * H;
W = W * diag(1./normW);

if nargout > 2
    INFO.E = E;
    INFO.UDtime = UDtime;
end


