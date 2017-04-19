function [ratios] = computeRegion(img)

    thresh = graythresh(img);
    BW = im2bw(img,thresh);
    
    s  = regionprops(BW,'centroid','Area','BoundingBox');
    [h,w,~] = size(img);
    
    count = numel(s);
    area = h*w;
    bbareas = zeros(count,1);
    ratios = cell(count,1);
    for i = 1:count
        boudingbox = s(i).BoundingBox;
        height = boudingbox(3);
        width = boudingbox(4);
        bbareas(i) = height*width/area;
        ratios{i} = s(i).Area/area;
    end
    ratios(bbareas==1) = [];
    ratios = cell2mat(ratios);
%     if isempty(ratios)
%         ratio = 0;
%     else
%         ratio = max(ratios);
%     end
end
    
    