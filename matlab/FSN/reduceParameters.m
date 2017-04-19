function reduceParameters()
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
    
    caffe_model_dir = '/home/fsn0924/train_with_single_stream_with_pool3_removed/';
    caffe_model_def = fullfile(caffe_model_dir,'vgg_16_net_reduced_by_zal_deploy.prototxt');
    caffe_weights_file = fullfile(caffe_model_dir,'vgg_16_net_reduced_by_zal.caffemodel');
    phase = 'test';
    net = caffe.Net(caffe_model_def,phase);
    net.copy_from(caffe_weights_file);
    
   model_dir = '/home/fsn0924/train_with_single_stream_with_pool3_removed';
   caffe_model_def = fullfile(model_dir,'pool3_removed_with_single_res_inception_module_deploy.prototxt');
   caffe_weights_file = fullfile(model_dir,'fixation_and_vgg_16_net.caffemodel');
   phase = 'test';
   vggNet = caffe.Net(caffe_model_def,caffe_weights_file,phase);
   
   fc_weights = vggNet.layers('fc_6').params(1).get_data();
   fc_bias = vggNet.layers('fc_6').params(2).get_data();
   
   fc6_weights = net.layers('fc6').params(1).get_data();
   fc6_bias = net.layers('fc6').params(2).get_data();
   
   [w,h,c,n] = size(fc6_weights);
   
   
%    fc_weights = reshape(fc_weights,[7,7,512,4096]);
%    fc_bias = reshape(fc_bias,size(fc6_bias));
   index = randi([1,4096],[1,256]);
   net.layers('fc6').params(1).set_data(fc_weights(:,:,:,index));
   net.layers('fc6').params(2).set_data(fc_bias(index));
   
   fc_weights = vggNet.layers('fc_7').params(1).get_data();
   fc_bias = vggNet.layers('fc_7').params(2).get_data();
   
   fc7_weights = net.layers('fc7').params(1).get_data();
   fc7_bias = net.layers('fc7').params(2).get_data();
%    
%    fc_weights = reshape(fc_weights,size(fc7_weights));
%    fc_bias = reshape(fc_bias,size(fc7_bias));
%    
   idx = randi([1,1024],[1,256]);
   net.layers('fc7').params(1).set_data(fc_weights(:,:,index,idx));
   net.layers('fc7').params(2).set_data(fc_bias(idx));
   
   net.save('./vgg_16_net_reduced_by_zal.caffemodel');
   disp(size(fc_weights));
   caffe.reset_all();
   