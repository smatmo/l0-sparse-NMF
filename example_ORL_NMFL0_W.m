%
% simple example application of NMFL0-W.
% The ORL data base is needed, which can be downloaded from
% http://www.cl.cam.ac.uk/Research/DTG/attarchive:pub/data/att_faces.tar.Z
%

clear all

% set this path to the folder which contains the folders s1-s40
ORLpath = '../orl_faces';

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

% rp=randperm(size(DataORL,2));
% figure;
% colormap(gray);
% for k=1:25
%     subplot(5,5,k)
%     imagesc(reshape(DataORL(:,rp(k)),112,92));
%     drawnow;
% end

[D,N] = size(DataORL);

prcntList = [33,25,10];

for prcntCount = 1:length(prcntList)
    prcnt = prcntList(prcntCount);
    
    options.K = 25;
    options.L = round(D*prcnt/100);
    options.numIter = 30;
    options.updateType = 'ANLS_FC';
    options.numUpdateIter = 10;
    
    [W,H,INFO] = NMFL0_W(DataORL,options);
    
    figure
    tmpW = repmat(max(W),size(W,1),1) - W;
    tmpW = tmpW - repmat(min(tmpW),size(tmpW,1),1);
    tmpW = tmpW * diag((1./max(tmpW)));
    imagesc(concatImg(tmpW,5,5,92,112,3))
    colormap(gray)
    axis off
    drawnow
end


