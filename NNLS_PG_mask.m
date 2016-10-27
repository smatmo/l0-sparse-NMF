function [H,grad,iter] = NNLS_PG_mask(V,W,Hinit,tol,maxiter,mask)
%
% Nonnegative Least-Squares using projected gradients
%
% Copyright (c) 2005-2008 Chih-Jen Lin
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
% 
% 1. Redistributions of source code must retain the above copyright
% notice, this list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
% 
% 3. Neither name of copyright holders nor the names of its contributors
% may be used to endorse or promote products derived from this software
% without specific prior written permission.
% 
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% H, grad: output solution and gradient
% iter: #iterations used
% V, W: constant matrices
% Hinit: initial solution
% tol: stopping tolerance
% maxiter: limit of iterations
%
%
%
% Modified by Robert Peharz: values in H(~mask) are constraint to be zero,
% to be used in a subroutine in the dictionary update stage for NMF with 
% ℓ0-constraints, described in
% Peharz and Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% Robert Peharz, 2011
%

H = Hinit; 
WtV = W'*V; 
WtW = W'*W;
numelH = numel(H);

if ~islogical(mask)
    mask=mask~=0;
end

alpha = 1; 
beta = 0.1;
for iter=1:maxiter   
    grad = WtW*H - WtV;
    grad = grad .* mask;
    projgrad = norm(grad(grad < 0 | H >0),'fro');
    if projgrad < tol
        break
    end
    
    % search step size
    for inner_iter=1:20
        Hn = max(H - alpha*grad, 0);
        d = Hn-H;
        gradd = grad(:)' * d(:);                  % gradd = sum(sum(grad.*d));
        dQd = reshape(WtW*d,1,numelH) * d(:);     % dQd = sum(sum((WtW*d).*d));
        suff_decr = 0.99*gradd + 0.5*dQd < 0;
        
        if inner_iter==1
            decr_alpha = ~suff_decr;
            Hp = H;
        end
        
        if decr_alpha
            if suff_decr
                H = Hn; 
                break;
            else
                alpha = alpha * beta;
            end
        else
            if ~suff_decr | Hp == Hn
                H = Hp; 
                break;
            else
                alpha = alpha/beta; 
                Hp = Hn;
            end
        end
    end
end

