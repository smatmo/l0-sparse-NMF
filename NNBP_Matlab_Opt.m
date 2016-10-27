function [h, exitflag] = NNBP_Matlab_Opt(W, x, e)
%
% solve the problem 
%
% minimize sum(h)
% s.t.     all(h >= 0)
%          norm(W*h - x)^2 <= e
%
% using Matlab's fmincon
%

%%% Set up the auxiliary data.
WtX = W'*x;
auxdata = {W, x, e, W'*W, WtX};

hlength = size(W, 2);

options = optimset('GradObj','on');
options = optimset(options, 'MaxIter', 10000);
options = optimset(options, 'MaxFunEvals', 10000);
%options = optimset(options, 'Display', 'iter-detailed');
% options = optimset(options, 'Display', 'iter');
options = optimset(options, 'Display', 'off');
%options = optimset(options, 'algorithm', 'trust-region-reflective');
options = optimset(options, 'algorithm', 'interior-point');
%options = optimset(options, 'algorithm', 'active-set');
%options = optimset(options, 'Hessian', 'lbfgs');
options = optimset(options, 'Hessian', 'user-supplied', 'HessFcn', @(h, lambda) hessian(h,lambda,auxdata));
% options = optimset(options, 'LargeScale', 'on');
options = optimset(options, 'UseParallel','always');
options = optimset(options, 'GradConstr', 'on');
%options = optimset(options, 'PrecondBandWidth', 10);
%options = optimset(options, 'TolPCG', 10);
options = optimset(options, 'ScaleProblem', 'obj-and-constr');
% options = optimset(options, 'ScaleProblem', 'none');
%options = optimset(options, 'MaxProjCGIter', 10);
%options = optimset(options, 'AlwaysHonorConstraints', 'none');
% options = optimset(options, 'SubproblemAlgorithm', 'cg');
%options = optimset(options, 'MaxPCGIter', 100);
%options = optimset(options, 'TolFun', 1e-1);
%options = optimset(options, 'TolX', 1e-1);
%options = optimset(options, 'TolCon', 1e-3);
% options = optimset(options, 'DerivativeCheck', 'on');
lb = zeros(hlength,1);
ub = [];

[h, ~, exitflag] = fmincon(@(h) objective(h, auxdata), zeros(hlength,1), [], [], [], [], lb, ub, @(h) constraints(h, auxdata), options);

function [f,g] = objective(h, auxdata)
[W, X, e, WtW, WtX] = deal(auxdata{:});

f = sum(h);
g = ones(length(h), 1);

function [c,ceq,J,Jeq] = constraints(h, auxdata)
[W, X, e, WtW, WtX] = deal(auxdata{:});

c = norm(W*h-X)^2 - e;
ceq = [];
J = sparse(2*(WtW*h-WtX));
Jeq = [];

function Hstruc = hessianstructure(auxdata)
[W, X, e, WtW, WtX] = deal(auxdata{:});

%Hstruc = sparse(tril(ones(150,150)));
Hstruc = sparse(tril(sparse(WtW)));


function H = hessian(h, lambda, auxdata)
[W, X, e, WtW, WtX] = deal(auxdata{:});

H = lambda.ineqnonlin(1)*2*WtW;


