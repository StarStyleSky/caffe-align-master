function generateMapList()
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
   
    caffe_model_Dir ='/media/yexiguafu/新加卷1/Experiment/SalObjectDetection/msra10k/';
    caffe_model_def = fullfile(caffe_model_Dir,'reduce_vgg_16_net_with_stacked_deploy.prototxt');
    caffe_weights_file = fullfile(caffe_model_Dir,'saliency_sum_fusion_msra10k_iter_45000.caffemodel');
    
    phase = 'test';
    net = caffe.Net(caffe_model_def,caffe_weights_file,phase);
    
    dirpath = '/media/yexiguafu/新加卷/Project02/SalObjectDetection/DUTOMRON/Image';
    desDir ='/media/yexiguafu/新加卷/Project02/SalObjectDetection/DUTOMRON/dilation_conv_with_sum_fusion';
    if ~exist(desDir,'dir')
        mkdir(desDir);
    end
    endings = {'.jpeg','.jpg','bmp','.png'};
    images = cellfun(@(x)dir([dirpath,filesep,'*',x]),endings,'UniformOutput',false);

    images = cat(1,images{:});
    count = size(images,1);
    tic;
    for i = 1:count
        filepath = fullfile(dirpath,images(i).name);
        fprintf('%d/%d:%s\n',i,count,filepath);
        [~,name,~] = fileparts(filepath);
        smap_file_path = fullfile(desDir,[name,'.png']);
        if ~exist(smap_file_path,'file')
            
            [smap] = computeSaliencyMap(net,filepath);
            imwrite(smap,smap_file_path);
            
        end
    end
    toc;
    caffe.reset_all();