function [smap] = postpreprocess(res,param)
    saliency_mean = 127;
    res = res.*128;
    res = res + saliency_mean;
    res(res<0) = 0;
    res(res>255) = 255;
    map = permute(map,[2,1,3]);
    map = imresize(res,[param.height,param.width],'bicubic');
    blurFilter = fspecial('gaussian',param.blurSize,param.sigma);
    
    smap = imfilter(map,blurFilter);
    smap = smap./max(smap(:))*255;
    smap = uint8(smap);
end