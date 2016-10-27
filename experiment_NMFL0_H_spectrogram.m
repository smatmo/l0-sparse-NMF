% reproduce results for
%
% R. Peharz and F. Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% section 4.2, NMFL0-H applied to spectrogram data, Figure 3.
%
% Attention: this will run long, since NMFL0-H is called 
% numel(Lrange) * numelKLrange) * 3 times, using 200 outer iterations.
%
% Robert Peharz, 2011
%

clear all

%%% path to audio data of speaker 1 in GRID corpus, which can be downloaded at 
%%% http://spandh.dcs.shef.ac.uk/gridcorpus/s1/audio/s1.tar
dataPath = '/afs/spsc.tugraz.at/resources/databases/corpusScss/training/1/';

%%% result path
resultPath = 'Results/NMFL0_H/';
if ~exist(resultPath,'dir')
    mkdir(resultPath);
end

%%% NMFL0 params
Lrange = [5,10,20];
Krange = [100,250,500];
numL = length(Lrange);
numK = length(Krange);
numIter = 200;

%%% Signal params
numSeconds = 120;
fsTarget = 8000;
windowLen = 512;
numOverlap = 256;
FFTlen = 512;

%%% allocate signal
x = zeros(numSeconds * fsTarget,1);

%%% Load Data
fileList = dir(dataPath);

%%% read in wav files and concatenate, until signal x is full
idx = 1;
for k=3:length(fileList)
    [x_,fsIn] = wavread([dataPath,fileList(k).name]);
    x_ = resample(x_,fsTarget, fsIn);
    numSamples = min(length(x) - idx + 1, length(x_));
    x(idx:idx+numSamples-1) = x_(1:numSamples);
    idx=idx+numSamples;
    if idx > length(x)
        break;
    end
end
if idx <= length(x)
    x = x(1:idx-1);
    fprintf('total audio data: %d seconds\n', length(x) / fsTarget);
end

%%% spectrogram
X = abs(spectrogram(x,windowLen, numOverlap, FFTlen));
frobX = norm(X,'fro');
EX = sum(X.^2);

%%% Run NMFL0-H
for Kcount = 1:numK
    K = Krange(Kcount);
    for Lcount = 1:numL
        L = Lrange(Lcount);
        
        fprintf('K: %d   L: %d\n',K,L);
        
        paramStr = ['K', int2str(K), '_L', int2str(L)];
        
        %%% Set common initial W
        rand('seed', K*1000000 + L);
        rp = randperm(size(X,2));
        initW = X(:,rp(1:K));
        
        
        %%%%%%%%%%%%
        %%% ANLS %%%
        %%%%%%%%%%%%
        
        clear options;
        
        options.K = K;
        options.L = L;
        options.numIter = numIter;
        options.W = initW;
        
        options.sparseCoder = @rsNNLS___;
        options.updateType = 'ANLS_FC';
        options.numUpdateIter = 10;
        
        fprintf('rsNNLS - ANLS 10\n')
        [W,H,INFO] = NMFL0_H(X, options);
        
        save([resultPath,'rsNNLS_ANLS10_',paramStr,'.mat'],'W','H','INFO','frobX','EX');
        
        ANLStime = INFO.UDtime;
        fprintf('mean ANLStime: %f\n',mean(ANLStime));
        
        
        %%%%%%%%%%%%
        %%% KSVD %%%
        %%%%%%%%%%%%
        
        clear options;
        
        options.K = K;
        options.L = L;
        options.numIter = numIter;
        options.W = initW;
        options.timeBudgetUpdate = ANLStime;
        
        options.sparseCoder = @rsNNLS___;
        options.updateType = 'NNKSVD';
        options.timeBudgetUpdate = ANLStime;
        
        fprintf('rsNNLS - NNKSVD\n')
        [W,H,INFO] = NMFL0_H(X, options);
        
        save([resultPath,'rsNNLS_NNKSVD_',paramStr,'.mat'],'W','H','INFO','frobX','EX');
        
        fprintf('mean KSVDtime: %f\n',mean(INFO.UDtime));
        
        
        %%%%%%%%%%
        %%% MU %%%
        %%%%%%%%%%
        
        clear options;
        
        options.K = K;
        options.L = L;
        options.numIter = numIter;
        options.W = initW;
        
        options.sparseCoder = @rsNNLS___;
        options.updateType = 'MU';
        options.timeBudgetUpdate = ANLStime;
        
        fprintf('rsNNLS - MU\n')
        [W,H,INFO] = NMFL0_H(X, options);
        
        save([resultPath,'rsNNLS_MU_',paramStr,'.mat'],'W','H','INFO','frobX','EX');
        
        fprintf('mean MUtime: %f\n',mean(INFO.UDtime));
    end
end


%%

numSettings = numK * numL;

plotStyle = {'--','-','--'};
plotColor = {[0,0,0],[1,0,0],[0,0,1]};
legendText = {'ANLS','MU','NNK-SVD'};

relE = zeros(numIter,numSettings,3);

counter = 1;
for Kcount=1:numK
    K=Krange(Kcount);
    for Lcount=1:numL
        L=Lrange(Lcount);
        paramStr = ['K', int2str(K), '_L', int2str(L)];
        
        load([resultPath,'rsNNLS_ANLS10_',paramStr,'.mat']);
        refE = INFO.E;
        relE(:,counter,1) = ones(numIter,1);
        load([resultPath,'rsNNLS_MU_',paramStr,'.mat']);
        relE(:,counter,2) = INFO.E ./ refE;
        load([resultPath,'rsNNLS_NNKSVD_',paramStr,'.mat']);
        relE(:,counter,3) = INFO.E ./ refE;
        
        counter = counter + 1;
    end
end

iter = [1:numIter];
meanRelE = squeeze(mean(relE,2));
stdRelE = squeeze(std(relE,1,2));

figure(1)
clf
hold on

plot(100*meanRelE(:,1),plotStyle{1},'color',plotColor{1});
for m=2:3
    errorbar(100*meanRelE(:,m),100*stdRelE(:,m),plotStyle{m},'color',plotColor{m});
end

box on
grid on
legend(legendText)
xlabel('Number of Iterations')
ylabel('relative RMSE [%]')


