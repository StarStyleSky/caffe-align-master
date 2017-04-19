function [smap] = computeSaliencyMap(net,impath)
    img = imread(impath);
    [h,w,c] = size(img);
    if c<3
        img = repmat(img,[1,1,3]);
    end
    mean_value = [103.939,116.779,123.68];     %BGR
    fixation_data = preprocess(img);
    
    image = imresize(img,[280,280]);
    img_ = single(image);
    img_ = img_(:,:,[3,2,1]);  %RGB——>BGR
    img_ = bsxfun(@minus,img_,reshape(mean_value,[1,1,3]));
    vgg_img = permute(img_,[2,1,3]);
    
    net.blobs('fixation').set_data(fixation_data);
    net.blobs('image').set_data(vgg_img);
    net.forward_prefilled();
    map = net.blobs('prob').get_data();
    map = permute(map,[2,1]);
    smap = imresize(map,[h,w]);

    smap = uint8(smap.*255);
    
%     backmap = net.blobs('probback').get_data();
%     backmap = permute(backmap,[2,1]);
%     backmap = imresize(backmap,[h,w]);
%     backmap = uint8(backmap.*255);
   
end