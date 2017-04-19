function addParameters_vgg16()
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
   
    model_dir = '/home/sogou/caffe-mask-rcnn/matlab/fsn';
    vgg16_caffe_model_dir = '/media/yexiguafu/新加卷1/CompresedFiles/models/VGG_ILSVRC_16_Layers';
    vgg16_weights_file = fullfile(vgg16_caffe_model_dir,'VGG_ILSVRC_16_layers_V1.caffemodel');
    caffe_model_def = fullfile(model_dir,'transition.prototxt');
    transition_weights_file = fullfile(model_dir,'transition.caffemodel');
    phase = 'test';
    net = caffe.Net(caffe_model_def,vgg16_weights_file,phase);
    net.save(transition_weights_file);
    
    caffe.reset_all();
    target_weights_file = fullfile(model_dir,'transition.caffemodel');
    target_def_file = fullfile(model_dir,'fully_convolutional_vgg16.prototxt');
    if ~exist(target_weights_file,'file')
        target_net = caffe.Net(target_def_file,phase);
        target_net.copy_from(transition_weights_file);
    else
        target_net = caffe.Net(target_def_file,target_weights_file,phase);
    end
    
    vgg16_model_dir = '/media/yexiguafu/新加卷1/CompresedFiles/models/VGG_ILSVRC_16_Layers';
    caffe_model_def = fullfile(vgg16_model_dir,'VGG_ILSVRC_16_layers_V1.prototxt');
    caffe_weights_file = fullfile(vgg16_model_dir,'VGG_ILSVRC_16_layers_V1.caffemodel');
    phase = 'test';
    vggNet = caffe.Net(caffe_model_def,phase);
    vggNet.copy_from(caffe_weights_file);
    

   
   fc_weights = vggNet.layers('fc6').params(1).get_data();
   fc_bias = vggNet.layers('fc6').params(2).get_data();
   
   conv_fc_weights = reshape(fc_weights,[7,7,512,4096]);
%    target_weights = target_net.layers('fc6').params(1).get_data();
%    target_bias = target_net.layers('fc6').params(2).get_data();
   
   index = randi([1,4096],[1,512]);
   target_net.layers('fc6').params(1).set_data(conv_fc_weights(:,:,:,index));
   target_net.layers('fc6').params(2).set_data(fc_bias(index));
   
   fc_weights = vggNet.layers('fc7').params(1).get_data();
   fc_bias = vggNet.layers('fc7').params(2).get_data();
   
   conv_fc_weights = reshape(fc_weights,[1,1,4096,4096]);
   idx = randi([1,4096],[1,512]);
   target_net.layers('fc7').params(1).set_data(conv_fc_weights(:,:,index,idx));
   target_net.layers('fc7').params(2).set_data(fc_bias(idx));
   
   desDir = '/home/sogou/caffe-mask-rcnn/matlab/fsn';
   file_path = fullfile(desDir,'fully_convolutional_vgg16_512.caffemodel');
   target_net.save(file_path);
   
   caffe.reset_all();
   