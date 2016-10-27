% reproduce results for
%
% R. Peharz and F. Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% section 4.1, nonnegative sparse coding, Figure 1 (SNR = inf) and
% Figure 2 (SNR = 10).
%
% Robert Peharz, 2011
%

clear all

%%% if result for NLARS shall be reproduced, please download the code from
%%% http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/5523/zip/imm5523.zip,
%%% copy the file NLARS.m into this folder, and set withNLARS = 1;
withNLARS = 0;

%%% if results for nonnegative basis pursuit shall be reproduced, set
%%% withNNBP = 1;
%%%
%%% ATTENTION: this runs long, since here we use the Matlab fmincon solver.
%%% You can speed this up, by replacing NNBP_Matlab_Opt in the file NNBP.m
%%% with a faster solver, which solves the convex problem (w.r.t. h)
%%%
%%% minimize sum(h)
%%% s.t.     all(h >= 0)
%%%          sum((W*h - x).^2) <= e
%%%
withNNBP = 0;

resultPath = 'Results/SparseCoder';
if ~exist(resultPath,'dir')
    mkdir(resultPath);
end

%%% number of turns, to be averaged over
numTurns = 10;

%%% dimensionality of data/dictionary
D = 100;

%%% number of data samples per turn
N = 100;

%%% OC: overcompleteness, dictionary size K = OC * D
OCrange = [2,4,8];
numOC = length(OCrange);

%%% sparseness factor (number of allowed nonzeros per column of H)
%%% it must be that max(Lrange) >= min(OCrange) * D
Lrange = [5:5:50];
numL = length(Lrange);

%%% SNR in dB
% for Figure 1
SNR = inf;
% for Figure 2
% SNR = 10;


%%%
numCorrectNMP = zeros(numTurns, numL, numOC);
numCorrectNNBP = zeros(numTurns, numL, numOC);
numCorrectSNNLS = zeros(numTurns, numL, numOC);
numCorrectRSNNLS = zeros(numTurns, numL, numOC);
numCorrectNLARS = zeros(numTurns, numL, numOC);

%%%
errorNMP = zeros(numTurns, numL, numOC);
errorNNBP = zeros(numTurns, numL, numOC);
errorSNNLS = zeros(numTurns, numL, numOC);
errorRSNNLS = zeros(numTurns, numL, numOC);
errorNLARS = zeros(numTurns, numL, numOC);

%%%
timeNMP = zeros(numTurns, numL, numOC);
timeNNBP = zeros(numTurns, numL, numOC);
timeSNNLS = zeros(numTurns, numL, numOC);
timeRSNNLS = zeros(numTurns, numL, numOC);
timeNLARS = zeros(numTurns, numL, numOC);

%%%
frobX = zeros(numTurns, numL, numOC);


for turn = 1:numTurns
    for OCcount = 1:numOC
        K=D*OCrange(OCcount);
        
        rand('seed', turn + (OCcount-1) * numTurns);
        randn('seed',turn + (OCcount-1) * numTurns);
        
        W = createDictionaryRand(D,K);
        fprintf('\n\n\nTurn: %d   OC: %d   coherence: %f\n',turn,OCrange(OCcount),max(max(W'*W-diag(diag(W'*W)))));
        
        for Lcount = 1:numL
            L = Lrange(Lcount);
            
            %%% generate true coding matrix
            Htrue = zeros(K,N);
            for n = 1:N
                rp = randperm(K);
                Htrue(rp(1:L),n) = 10*abs(randn(L,1));
            end
            
            %%% generate true data + noise with desired SNR
            X = W*Htrue;
            if SNR < inf
                Noise = rand(D,N);
                EN = sum(Noise.^2);
                EX = sum(X.^2);
                Noise = Noise * diag(sqrt((EX ./ (10^(SNR/10)*EN))));
                X = X + Noise;
            end
            frobX(turn, Lcount, OCcount) = norm(X,'fro');
            
            fprintf('L: %d\n',L);
            
            if 1
                %%% the sparse coder from the MLSP paper
                fprintf('NMP');
                tic;
                
                H = NMP(X,W,[],L);
                
                timeNMP(turn, Lcount, OCcount) = toc;
                errorNMP(turn, Lcount, OCcount) = norm(X-W*H,'fro');
                numCorrectNMP(turn, Lcount, OCcount) = mean(sum((H>0) & (Htrue>0)));
                fprintf('\t... t: %f \tE: %f \tCorrect: %f\n',timeNMP(turn, Lcount, OCcount), errorNMP(turn, Lcount, OCcount), numCorrectNMP(turn, Lcount, OCcount));
                save([resultPath,'/Result_NMP_',sprintf('SNR%d',SNR),'.mat'],'timeNMP','errorNMP','numCorrectNMP','frobX');
            end
            
            if withNNBP
                %%% here we use the (slow) Matlab implementation, since most
                %%% people will not have IPOpt...
                %%% attention: this will run long
                fprintf('NNBP');
                tic;
                
                if SNR == inf
                    H = NNBP(X,W,L);
                else
                    H = NNBP(X,W,L,sum(X.^2) / 10^((SNR+9) / 10));
                end
                
                timeNNBP(turn, Lcount, OCcount) = toc;
                errorNNBP(turn, Lcount, OCcount) = norm(X-W*H,'fro');
                numCorrectNNBP(turn, Lcount, OCcount) = mean(sum((H>0) & (Htrue>0)));
                fprintf('\t... t: %f \tE: %f \tCorrect: %f\n',timeNNBP(turn, Lcount, OCcount), errorNNBP(turn, Lcount, OCcount), numCorrectNNBP(turn, Lcount, OCcount));
                save([resultPath,'/Result_NNBP_',sprintf('SNR%d',SNR),'.mat'],'timeNNBP','errorNNBP','numCorrectNNBP','frobX');
            end
            
            if 1
                %%% sparse NNLS
                fprintf('sNNLS')
                tic;
                
                H = sparseNNLS(X,W,[],[],L,L);
                
                timeSNNLS(turn, Lcount, OCcount) = toc;
                errorSNNLS(turn, Lcount, OCcount) = norm(X-W*H,'fro');
                numCorrectSNNLS(turn, Lcount, OCcount) = mean(sum((H>0) & (Htrue>0)));
                fprintf('\t... t: %f \tE: %f \tCorrect: %f\n',timeSNNLS(turn, Lcount, OCcount), errorSNNLS(turn, Lcount, OCcount), numCorrectSNNLS(turn, Lcount, OCcount));
                save([resultPath,'/Result_SNNLS_',sprintf('SNR%d',SNR),'.mat'],'timeSNNLS','errorSNNLS','numCorrectSNNLS','frobX');
            end
            
            if 1
                %%% reverse sparse NNLS
                fprintf('rsNNLS')
                tic;
                
                H = sparseNNLS(X,W,[],[],L,K);
                
                timeRSNNLS(turn, Lcount, OCcount) = toc;
                errorRSNNLS(turn, Lcount, OCcount) = norm(X-W*H,'fro');
                numCorrectRSNNLS(turn, Lcount, OCcount) = mean(sum((H>0) & (Htrue>0)));
                fprintf('\t... t: %f \tE: %f \tCorrect: %f\n',timeRSNNLS(turn, Lcount, OCcount), errorRSNNLS(turn, Lcount, OCcount), numCorrectRSNNLS(turn, Lcount, OCcount));
                save([resultPath,'/Result_RSNNLS_',sprintf('SNR%d',SNR),'.mat'],'timeRSNNLS','errorRSNNLS','numCorrectRSNNLS','frobX');
            end
            
            if withNLARS
                %%% NLARS
                warning('off','MATLAB:nearlySingularMatrix');
                
                fprintf('NLARS')
                H = zeros(K,N);
                WtW = W'*W;
                tic;
                
                for n = 1:N
                    [beta, path, lambda] = NLARS(WtW, W' * X(:,n), 1);
                    path = path(:, sum(path > 0) <= L);
                    H(:,n) = path(:,end);
                end
                
                timeNLARS(turn, Lcount, OCcount) = toc;
                errorNLARS(turn, Lcount, OCcount) = norm(X-W*H,'fro');
                numCorrectNLARS(turn, Lcount, OCcount) = mean(sum((H>0) & (Htrue>0)));
                fprintf('\t... t: %f \tE: %f \tCorrect: %f\n',timeNLARS(turn, Lcount, OCcount), errorNLARS(turn, Lcount, OCcount), numCorrectNLARS(turn, Lcount, OCcount));
                save([resultPath,'/Result_NLARS_',sprintf('SNR%d',SNR),'.mat'],'timeNLARS','errorNLARS','numCorrectNLARS','frobX');
                
                warning('on','MATLAB:nearlySingularMatrix');
            end
        end
    end
end


%%

%%% Plot results
K = OCrange * D;

% load('Results/SparseCoder/Result_NMP.mat')
% load('Results/SparseCoder/Result_NNBP.mat')
% load('Results/SparseCoder/Result_SNNLS.mat')
% load('Results/SparseCoder/Result_RSNNLS.mat')
% load('Results/SparseCoder/Result_NLARS.mat')

plotStyle = {'-g*', '-rs', '-kv', '-bo', '-m^'};

if withNNBP
    legendText = {'NMP','NNBP', 'sNNLS', 'rsNNLS', 'NLARS'};
else
    legendText = {'NMP', 'sNNLS', 'rsNNLS', 'NLARS'};
end

figure(1)
clf

for k = 1:length(K)
    
    %%% SNR
    subplot(3,length(K), k)
    hold on
    
    y = 10*log10(squeeze(mean( frobX(:,:,k).^2./errorNMP(:,:,k).^2 ,1)));
    plot(Lrange, y, plotStyle{1})
    if withNNBP
        y = 10*log10(squeeze(mean( frobX(:,:,k).^2./errorNNBP(:,:,k).^2 ,1)));
        plot(Lrange, y, plotStyle{2})
    end
    y = 10*log10(squeeze(mean( frobX(:,:,k).^2./errorSNNLS(:,:,k).^2 ,1)));
    plot(Lrange, y, plotStyle{3})
    y=10*log10(squeeze(mean(  frobX(:,:,k).^2./errorRSNNLS(:,:,k).^2  ,1)));
    plot(Lrange, y, plotStyle{4})
    if withNLARS
        y=10*log10(squeeze(mean(  frobX(:,:,k).^2./errorNLARS(:,:,k).^2  ,1)));
        plot(Lrange, y, plotStyle{5})
    end
    
    box on
    grid on
    if k==1
        ylabel('SNR [dB]')
    end
    title(['Number of basis vectors: ',num2str(K(k))])
    
    
    %%% number correctly identified dictionary vectors
    subplot(3,length(K), k + 3)
    hold on
    
    y = 100*squeeze(mean(numCorrectNMP(:,:,k) ./ repmat(Lrange,[numTurns,1,1]),1));
    plot(Lrange, y, plotStyle{1});
    if withNNBP
        y = 100*squeeze(mean(numCorrectNNBP(:,:,k) ./ repmat(Lrange,[numTurns,1,1]),1));
        plot(Lrange, y, plotStyle{2})
    end
    y = 100*squeeze(mean(numCorrectSNNLS(:,:,k) ./ repmat(Lrange,[numTurns,1,1]),1));
    plot(Lrange, y, plotStyle{3})
    y = 100*squeeze(mean(numCorrectRSNNLS(:,:,k) ./ repmat(Lrange,[numTurns,1,1]),1));
    plot(Lrange, y, plotStyle{4})
    if withNLARS
        y = 100*squeeze(mean(numCorrectNLARS(:,:,k) ./ repmat(Lrange,[numTurns,1,1]),1));
        plot(Lrange, y, plotStyle{5})
    end
    
    box on
    grid on
    
    if k==1
        ylabel('% correct')
    end
    if k==3
        legend(legendText);
    end
    
    
    %%% time
    subplot(3,length(K), k + 6)
    
    semilogy(Lrange, squeeze(mean(timeNMP(:,:,k),1)), plotStyle{1})
    hold on
    if withNNBP
        semilogy(Lrange, squeeze(mean(timeNNBP(:,:,k),1)), plotStyle{2})
    end
    semilogy(Lrange, squeeze(mean(timeSNNLS(:,:,k),1)), plotStyle{3})
    semilogy(Lrange, squeeze(mean(timeRSNNLS(:,:,k),1)), plotStyle{4})
    if withNLARS
        semilogy(Lrange, squeeze(mean(timeNLARS(:,:,k),1)), plotStyle{5})
    end
    
    grid on
    
    xlabel('L (# non-zero coefficients)')
    if k==1
        ylabel('time [s]')
    end
end


