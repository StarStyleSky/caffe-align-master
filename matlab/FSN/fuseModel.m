function fuseModel()
    addpath(genpath('..'));
    caffe.reset_all();
    use_gpu = 0;
    if exist('use_gpu','var') && use_gpu==1
        caffe.set_mode_gpu();
        gpu_id = 0;  % Use the last GPU for manual validation
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
    
    caffe_model_def = './models/deep_fixation_and_vgg_16_net.prototxt';
    caffe_weights_vgg = '/media/yexiguafu/新加卷/CompresedFiles/models/VGG_ILSVRC_16_Layers/VGG_ILSVRC_16_layers_V1.caffemodel';
    caffe_weights_deep = '/media/yexiguafu/新加卷/CompresedFiles/models/deep_fixation_net/deep_net_model.caffemodel';
    phase = 'test';
    net = caffe.Net(caffe_model_def,phase);
    net.copy_from(caffe_weights_vgg);
    net.copy_from(caffe_weights_deep);
    net.save('./fixation_and_vgg_16_net.caffemodel');
    caffe.reset_all();
    
    