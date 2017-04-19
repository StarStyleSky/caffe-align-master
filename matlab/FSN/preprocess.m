function [img] = preprocess(image)
    input_size = [280,280];
    mean_value = [110,110,118];
    input_scale = 0.0078431372549;

    [~,~,channels] = size(image);
    if channels < 3
        image = repmat(image,[1,1,3]);        
    end
    image = imresize(image,input_size);
    
    img_ = single(image);
    img_ = img_(:,:,[3,2,1]);
    img_ = bsxfun(@minus,img_,reshape(mean_value,[1,1,3]));
    img_ = bsxfun(@times,img_,input_scale);
    img = permute(img_,[2,1,3]);
end