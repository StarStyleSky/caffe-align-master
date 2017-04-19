function test_network()
     addpath(genpath('/home/sogou/caffe-mask-rcnn/matlab'));
    caffe.reset_all();
    use_gpu = 1;
    if exist('use_gpu','var') && use_gpu==1
        caffe.set_mode_gpu();
        gpu_id = 0;  % Use the last GPU for manual validation
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
   
    caffe_model_Dir ='/home/sogou/fsn';
    caffe_model_def = fullfile(caffe_model_Dir,'feature_assemble_fsn_deploy.prototxt');
    caffe_weights_file = fullfile(caffe_model_Dir,'fully_convolutional_vgg16_512.caffemodel');
%     suffix = 'dual_streams_pre_init_single_inception_v2_0312_iteration_45000';
    phase = 'test';
    if ~(exist(caffe_model_def,'file') && exist(caffe_weights_file,'file'))
        fprintf('Caffe models does not exist,please check file first.\n');
        return ;
    end
    net = caffe.Net(caffe_model_def,caffe_weights_file,phase);
    caffe.reset_all();