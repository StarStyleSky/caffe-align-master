function demo()
    addpath(genpath('..'));
    caffe.reset_all();
    
    modelDir = '/media/yexiguafu/yexiguafuqihao/BaiduNetdiskDownload/myNet3';
    prototxt = fullfile(modelDir,'deploy.prototxt');
    caffemodel = fullfile(modelDir,'models/myNet3_iter_120000.caffemodel');
    if ~(exist(prototxt,'file') && exist(caffemodel,'file'))
        fprintf('model file does not exist.\n');
        return;
        endc
    
    use_gpu = 1;
    phase = 'test';
    if exist('use_gpu','var') && use_gpu == 1
        caffe.set_mode_gpu();
        device_id = 0;
        caffe.set_device(device_id);
    else
        caffe.set_mode_cpu();
    end
    
    net = caffe.Net(prototxt,caffemodel,phase);
    fprintf('Loading caffe model is done.\n');
    
    impath = '/media/yexiguafu/yexiguafuqihao/BaiduNetdiskDownload/myNet2/CityScapes/val_data/frankfurt_000000_001236_1_leftImg8bit.png';
    mean_file = '/home/sogou/myNet02/data/train_data.binaryproto';
    
    im_size = 512;
    mean_data = caffe.io.read_mean(mean_file);
    im = imread(impath);
    im = single(im);
    im = imresize(im,[im_size,im_size]);
    im = im(:,:,[3,2,1]) - mean_data;
    im = permute(im,[2,1,3]);
    
    net.blobs('data').set_data(im);
    net.forward_prefilled();
    prob = net.blobs('prob').get_data();
    im_size = 512;
    prob = imresize(prob,[im_size,im_size]);
    filepath = 'visualizationCode/colormapcs.mat';
    load(filepath);
    [~,imPred] = max(prob,[],3);
    caffe.reset_all();
end
   