function addParameters()
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
    model_dir = '/media/yexiguafu/新加卷/CompresedFiles/models/vgg16_reduced_yexiguafu/';
    caffe_model_def = fullfile(model_dir,'transition.prototxt');
    caffe_weights_file = fullfile(model_dir,'transition.caffemodel');
    phase = 'test';
    net = caffe.Net(caffe_model_def,caffe_weights_file,phase);
    
    
    caffe_model_dir = '/media/yexiguafu/新加卷/CompresedFiles/models/VGG_ILSVRC_16_Layers';
    caffe_model_def = fullfile(caffe_model_dir,'VGG_ILSVRC_16_layers_V1.prototxt');
    caffe_weights_file = fullfile(caffe_model_dir,'VGG_ILSVRC_16_layers_V1.caffemodel');
    phase = 'test';
    vggNet = caffe.Net(caffe_model_def,phase);
    vggNet.copy_from(caffe_weights_file);
    

   
   fc_weights = vggNet.layers('fc6').params(1).get_data();
   fc_bias = vggNet.layers('fc6').params(2).get_data();
   
%    fc6_weights = net.layers('fc6').params(1).get_data();
%    fc6_bias = net.layers('fc6').params(2).get_data();
   
%    fc_weights = reshape(fc6_weights,[7,7,512,4096]);
%    fc_bias = reshape(fc_bias,size(fc6_bias));
   index = randi([1,4096],[1,1024]);
   net.layers('fc6').params(1).set_data(fc_weights(:,index));
   net.layers('fc6').params(2).set_data(fc_bias(index));
   
   fc_weights = vggNet.layers('fc7').params(1).get_data();
   fc_bias = vggNet.layers('fc7').params(2).get_data();
   
%    fc7_weights = net.layers('fc_7').params(1).get_data();
%    fc7_bias = net.layers('fc_7').params(2).get_data();
   
%    fc_weights = reshape(fc_weights,size(fc7_weights));
%    fc_bias = reshape(fc_bias,size(fc7_bias));
   idx = randi([1,4096],[1,1024]);
   net.layers('fc7').params(1).set_data(fc_weights(index,idx));
   net.layers('fc7').params(2).set_data(fc_bias(idx));
   
   desDir = '/media/yexiguafu/新加卷/CompresedFiles/models/vgg16_reduced_yexiguafu';
   file_path = fullfile(desDir,'vgg16_reduced_yexiguafuqihao.caffemodel');
   vggNet.save(file_path);
%    disp(size(fc_weights));
   caffe.reset_all();
   