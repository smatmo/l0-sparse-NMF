function H = combinatorialLS(WtW, WtX, Pset)
%
% minimize ||X - W*H||_2 w.r.t. H(~Pset) = 0 using using a fast
% combinatorial approach. This is a modified version of the sub-routine
% cssls(CtC, CtA, Pset), taken from
%
% M. H. Van Benthem and M. R. Keenan, "Fast algorithm for the solution of
% large-scale non-negativity-constrained least squares problems", Journal
% of Chemometrics, 2004; 18: 441-450.
%
% The original function cssls encodes the Pset as
% codedPset = 2.^(lVar-1:-1:0)*Pset;
% However, for a large number of variables this makes numerical problems,
% and it can happen that two Psets [1,0,0,0,...,0,1,0]' and
% [1,0,0,0,...,0,0,1]' will be treated as the same set.
%
% in this version we use sortrows(Pset') instead of sort(codedPset), wich
% fixes this problem.
%
% Robert Peharz, 2011
%

H = zeros(size(WtX));
if (nargin == 2) || isempty(Pset) || all(Pset(:))
    H = WtW\WtX;
else
    [~, numRHS] = size(Pset);
    [sortedPset, sortedEset] = sortrows(Pset');
    sortedPset = sortedPset';
    breaks = sum(abs(diff(sortedPset,1,2)));
    breakIdx = [0 find(breaks) numRHS];
    for k = 1:length(breakIdx)-1
        colIdx = sortedEset(breakIdx(k)+1:breakIdx(k+1));
        varIdx = Pset(:,sortedEset(breakIdx(k)+1));
        H(varIdx,colIdx) = WtW(varIdx,varIdx) \ WtX(varIdx,colIdx);
    end
end
