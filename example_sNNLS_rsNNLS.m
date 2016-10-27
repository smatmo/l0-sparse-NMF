%
% simple example application of sNNLS and rsNNLS
%

clear all

%%% data dimensionality
D = 100;

%%% number data samples
N = 100;

%%% OC: overcompleteness
OCrange = [1,2,4,8];
numOC = length(OCrange);
Lrange = [5:5:50];
numL = length(Lrange);

for OCcount = 1:numOC
    K = D*OCrange(OCcount);
    
    W = createDictionaryRand(D,K);
    fprintf('\n\n\novercompleteness: %d   coherence: %f\n',OCrange(OCcount),max(max(W'*W-diag(diag(W'*W)))));
    
    for Lcount = 1:numL
        L = Lrange(Lcount);
        fprintf('L: %d\n',L);
        
        %%% make "true" coding matrix
        Htrue = zeros(K,N);
        for n = 1:N
            rp = randperm(K);
            Htrue(rp(1:L),n) = 10*abs(randn(L,1));
        end
        
        %%% make synthetic data
        X = W*Htrue;
        frobX = norm(X,'fro');
        
        %%% run sparse coders
        fprintf('sNNLS')
        tic
        H = sparseNNLS(X,W,[],[],L,L);
        fprintf('\t... t: %f \tE: %f \tCorrect: %f\n', toc, norm(X-W*H,'fro'), mean(sum((H>0) & (Htrue>0))));
        
        fprintf('rsNNLS')
        tic
        H = sparseNNLS(X,W,[],[],L,size(W,2));
        fprintf('\t... t: %f \tE: %f \tCorrect: %f\n', toc, norm(X-W*H,'fro'), mean(sum((H>0) & (Htrue>0))));
    end
end


