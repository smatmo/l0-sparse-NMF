% reproduce results for
%
% R. Peharz and F. Pernkopf, "Sparse nonnegative matrix factorization with
% ℓ0-constraints", Neurocomputing, 2012.
%
% section 4.3, NMFL0-W applied to face images, Figure 4.
%
% Robert Peharz, 2011
%

clear all

%%% path to ORL data base
%%% the data base can be downloaded from
%%% http://www.cl.cam.ac.uk/Research/DTG/attarchive:pub/data/att_faces.tar.Z
ORLpath = '../orl_faces';

%%% result path
resultPath = 'Results/NMFL0_W/';
if ~exist(resultPath,'dir')
    mkdir(resultPath);
end

%%% number of random restarts
numTurns = 10;

%%% list of l0-sparseness values (in percent)
prcntList = [33,25,10];

%%% If results for l1-sparse NMF by
%%%
%%% P. Hoyer, "Non-negative Matrix Factorization with Sparseness Constraints",
%%% Journal of Machine Learning Research 5, 2004,
%%%
%%% shall be reprouduced, please download the code available under
%%% http://www.cs.helsinki.fi/u/phoyer/software.html
%%% and extract it to some folder.
%%%
%%% The file nmfsc.m needs a slight modification since the original
%%% code has an infinite loop and never terminates; so please modify the first
%%% line of nmfsc.m into
%%%
%%%    function [W,H] = nmfsc(V, rdim, sW, sH, fname, showflag, numIter)
%%%
%%% , i.e. introduce the new parameter numIter, and modify line 66 into
%%%
%%%    while iter <= numIter
%%%
%%% , i.e. introduce a stopping criterion.
%%%
%%% Finally, set the following line to withL1NMF=1;
%%% and set L1NMFpath to the path where you placed the code.
%%%
%%% Note: In order to make a fair comparison, I additionally removed some
%%% unnecessary features in the nmfsc code, such as the intermediate saving
%%% of results, and the progress plots.
%%% In short, I removed lines 71-93.
%%%
withL1NMF = 1;
L1NMFpath = '../nmfpack/code';
if withL1NMF
    addpath(L1NMFpath)
end


%%% ----------------------------------------------------- %%%

%%% load ORL data base
DataORL = [];
for s = 1:40
    list = dir([ORLpath,'/s',num2str(s)]);
    if isempty(list)
        error([ORLpath, ' seems to be empty.']);
    end
    for k = 3:length(list)
        im = imread([ORLpath,'/s',num2str(s),'/',list(k).name ]);
        im = double(im);
        DataORL = [DataORL, im(:)];
    end
end

%%% display faces
% rp=randperm(size(DataORL,2));
% figure;
% colormap(gray);
% for k = 1:10
%     subplot(5,5,k)
%     imagesc(reshape(DataORL(:,rp(k)),112,92));
%     drawnow;
% end

[D,N] = size(DataORL);

for turn = 1:numTurns
    fprintf('Turn %d/%d\n',turn,numTurns);
    rand('state',turn);
    randn('state',turn);
    
    HoyerSparse = [];
    
    for prcntCount = 1:length(prcntList)
        prcnt = prcntList(prcntCount);
        
        options.K = 25;
        options.L = round(D*prcnt/100);
        options.numIter = 30;
        options.updateType = 'ANLS_FC';
        options.numUpdateIter = 10;
        
        c1 = clock;
        [W,H,INFO] = NMFL0_W(DataORL,options);
        TL0 = etime(clock,c1);
        
        ResultL0{prcntCount}.W = W;
        ResultL0{prcntCount}.H = H;
        ResultL0{prcntCount}.INFO = INFO;
        ResultL0{prcntCount}.TL0 = TL0;
        HoyerSparse = [HoyerSparse, mean(hoyerS(W))];
    end
    
    for sCount = 1:length(HoyerSparse)
        s = HoyerSparse(sCount);
        
        c1 = clock;
        [W,H] = nmfsc(DataORL, 25, s, [], 'NMFl1out', 0, 2500);
        TL1 = etime(clock,c1);
        
        ResultL1{sCount}.W = W;
        ResultL1{sCount}.H = H;
        ResultL1{sCount}.TL1 = TL1;
    end
    
    save([resultPath,'NMFFacesResult_Turn',int2str(turn),'.mat'],'ResultL0','ResultL1','HoyerSparse');
end

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% evaluate/plot Results %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SNRL0 = zeros(numTurns,length(prcntList));
SNRL1 = zeros(numTurns,length(prcntList));
SNRstar = zeros(numTurns,length(prcntList));

L0normL0 = zeros(numTurns,length(prcntList));
L0normL1 = zeros(numTurns,length(prcntList));

L1normL0 = zeros(numTurns,length(prcntList));
L1normL1 = zeros(numTurns,length(prcntList));

timeL0 = zeros(numTurns,length(prcntList));
timeL1 = zeros(numTurns,length(prcntList));

%%% nmfsc.m normalizes the data; the SNR has to be calculated accordingly
normData = DataORL / max(DataORL(:));

HSparse = zeros(numTurns,length(prcntList));

for k=1:numTurns
    load([resultPath,'NMFFacesResult_Turn',int2str(k),'.mat']);
    
    HSparse(k,:) = HoyerSparse;
    
    for l=1:3
        L = round(size(ResultL1{l}.W,1)*prcntList(l)/100);
        
        SNRL0(k,l) = norm(DataORL,'fro')^2 / norm(DataORL - ResultL0{l}.W * ResultL0{l}.H,'fro')^2;
        SNRL1(k,l) = norm(normData,'fro')^2 / norm(normData - ResultL1{l}.W * ResultL1{l}.H,'fro')^2;
        
        L0normL0(k,l) = mean(sum(ResultL0{l}.W > 0));
        L0normL1(k,l) = mean(sum(ResultL1{l}.W > 0));
        
        L1normL0(k,l) = mean(hoyerS(ResultL0{l}.W));
        L1normL1(k,l) = mean(hoyerS(ResultL1{l}.W));
        
        timeL0(k,l) = ResultL0{l}.TL0;
        timeL1(k,l) = ResultL1{l}.TL1;
        
        %%% prune smallest values of the l1-sparse basis vectors, to obtain
        %%% SNR* (see table 1 in the paper)
        W = ResultL1{l}.W;
        H = ResultL1{l}.H;
        for j = 1:size(W,2)
            [sL,sIdx] = sort(W(:,j),'descend');
            W(sIdx(L+1:end),j) = 0;
        end
        SNRstar(k,l) = norm(normData,'fro')^2 / norm(normData - W * H,'fro')^2;
    end
end

fprintf('\n\n\n');

for l=1:3
    fprintf('l1-NMF:  l0-sparseness: %3.3f %%   l1-sparseness: %3.3f   SNR: %3.3f dB   SNR*(dB): %3.3f dB   time: %10.3fs\n', 100*(mean(L0normL1(:,l),1) / D), mean(L1normL1(:,l),1), ...
        10 * log10(mean(SNRL1(:,l))), 10 * log10(mean(SNRstar(:,l))), mean(timeL1(:,l)));
    
    fprintf('NMFL0:   l0-sparseness: %3.3f %%   l1-sparseness: %3.3f   SNR: %3.3f dB                        time: %10.3fs\n', 100*(mean(L0normL0(:,l),1) / D), mean(L1normL0(:,l),1), ...
        10 * log10(mean(SNRL0(:,l))), mean(timeL0(:,l)));
    
    fprintf('\n\n');
end

for l=1:3
    figure(l)
    clf
    W = ResultL0{l}.W;
    W = repmat(max(W),size(W,1),1) - W;
    W = W - repmat(min(W),size(W,1),1);
    W = W * diag((1./max(W)));
    imagesc(concatImg(W,5,5,92,112,3))
    colormap(gray)
    axis off
    title(sprintf('NMFL0   L0: %3.2f',prcntList(l)))
    
    figure(3+l)
    W = ResultL1{l}.W;
    W = repmat(max(W),size(W,1),1) - W;
    W = W - repmat(min(W),size(W,1),1);
    W = W * diag((1./max(W)));
    imagesc(concatImg(W,5,5,92,112,3))
    colormap(gray)
    axis off
    title(sprintf('l1-NMF   L0: %3.2f%%',prcntList(l)))
end


