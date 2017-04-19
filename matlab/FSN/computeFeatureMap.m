function computeFeatureMap()
    addpath(genpath('..'));
    
   
    caffe.reset_all();
    use_gpu = 1;
    if exist('use_gpu','var') && use_gpu==1
        caffe.set_mode_gpu();
        gpu_id = 0;  % Use the last GPU for manual validation
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
    
    caffe_model_Dir ='./model';
    caffe_model_def = fullfile(caffe_model_Dir,'VGG_ILSVRC_16_layers_V1.prototxt');
    caffe_weights_file = fullfile(caffe_model_Dir,'VGG_ILSVRC_16_layers_V1.caffemodel');
    
    phase = 'test';
    net = caffe.Net(caffe_model_def,caffe_weights_file,phase);
    
    impath = '/home/yexiguafu/Downloads/Image/00000155.png';
    img = imread(impath);
    [h,w,c] = size(img);
    if c<3
        img = repmat(img,[1,1,3]);
    end
    mean_value = [103.939,116.779,123.68];
    image = imresize(img,[280,280]);
    img_ = single(image);
    img_ = img_(:,:,[3,2,1]);
    img_ = bsxfun(@minus,img_,reshape(mean_value,[1,1,3]));
    vgg_img = permute(img_,[2,1,3]);
    
    net.blobs('data').set_data(vgg_img);
    net.forward_prefilled();
    feat = net.blobs('conv5_3').get_data();
    disp(size(feat));
    