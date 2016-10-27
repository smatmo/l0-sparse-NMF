function H = NNBP(X,W,L,e)
%
% H = NNBP(X,W,L,e)
%
% For each column of X, solve nonnegative basis pursuit, select the largest
% L coefficients, and solve nonnegative least squares using the corresponding
% selected dictionary vectors.
%
% input:
%    X: D x N data matrix
%    W: D x K dictionary matrix
%    L: maximal number of nonzeros per column of H
%    e: vector of length N. 
%       maximal error per column (i.e. norm(x(:,n) - W * h(:,n))^2 < e(n)).
%       If problem turns out to be infeasible for n, e(n) <- 2*e(n) is
%       repeated until problem is feasible
%

[D,N]=size(X);
[DW,K]=size(W);
if D~=DW
    error('Dimensions are inconsistent')
end

if nargin < 4
    e = sum(X.^2) * 1e-12;
end

for n=1:N
    %%% solve nonnegative basis pursuit problem
    [H(:,n),exitflag] = NNBP_Matlab_Opt(W, X(:,n), e(n));
    
    %%% if infeasible, relax the error constraint by 3dB and repeat
    if exitflag < 0
        fprintf('!');        
        while 1
            e(n) = e(n) * 2;
            [H(:,n),exitflag] = NNBP_Matlab_Opt(W, X(:,n), e(n));
            if exitflag >= 0
                break
            end
        end
    end
    %%%
end

%%% set all but the L largest coefficients in each column of H to zero
[~,idx] = sort(H,'ascend');
for k=1:N
    H(idx(1:K-L,k),k) = 0;
end

%%% solve NNLS with selected coefficients
H = maskNNLS(X,W,[],[],H > 0);
