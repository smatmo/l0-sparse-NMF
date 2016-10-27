function img = concatImg(X,W,H,w,h,gw)
% img = concatImg(X,W,H,w,h,gw)
% 
% Rearange and concatenate the images stored in the columns of X
%

numImg = size(X,2);

img = ones((h+gw)*H+gw,(w+gw)*W+gw);
img= img*(max(X(:))-min(X(:))) / 2;

imgC = 1;
for l=1:W
    for k=1:H
        patch = reshape(X(:,imgC),h,w);
        img((k-1)*(h+gw)+(1+gw):(k-1)*(h+gw)+(h+gw), (l-1)*(w+gw)+(1+gw):(l-1)*(w+gw)+(w+gw)) = patch;
        
        if imgC == numImg
            break
        end
        imgC=imgC+1;
    end
end

