function testModel()
    addpath(genpath('../'));
    caffe.reset_all();
    use_gpu = 1;
    if exist('use_gpu','var') && use_gpu==1
        caffe.set_mode_gpu();
        gpu_id = 0;  % Use the last GPU for manual validation
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
    
    caffe_model_def = '/media/yexiguafu/新加卷/Experiment/SalObjectDetection/msra10k/revolution_net_ultimate/revoltion_net_deploy.prototxt';
    caffe_weights_file = '/media/yexiguafu/新加卷/Experiment/SalObjectDetection/msra10k/revolution_net_ultimate/revolution.caffemodel';
  
    phase = 'test';
    net = caffe.Net(caffe_model_def,phase);
    net.copy_from(caffe_weights_file);
    
    impath = '/home/yexiguafu/Downloads/Image/110932821430.png';
    img = imread(impath);
    img = imresize(img,[280,280]);
    [width,height,channels,count] = size(img);
    net.blobs('data').reshape([width,height,channels,count]);
    net.reshape();
    net.blobs('data').set_data(img);
    net.forward_prefilled();
    res = net.blobs('prob').get_data();
    disp(size(res));