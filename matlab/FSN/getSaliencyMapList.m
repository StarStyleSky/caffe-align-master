function getSaliencyMapList()
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
   
    caffe_model_Dir ='/home/sogou/fsn';
    caffe_model_def = fullfile(caffe_model_Dir,'feature_assemble_fsn_deploy.prototxt');
    caffe_weights_file = fullfile(caffe_model_Dir,'sigmoid_constrained_feature_assemble_fsn_0413_iter_45000.caffemodel');
    suffix = 'sigmoid_constrained_feature_assemble_fsn_0413_iter_45000';
    phase = 'test';
    if ~(exist(caffe_model_def,'file') && exist(caffe_weights_file,'file'))
        fprintf('Caffe models does not exist,please check file first.\n');
        return ;
    end
    net = caffe.Net(caffe_model_def,caffe_weights_file,phase);

    dirpath = '/media/yexiguafu/新加卷1/Project02/SalObjectDetection';
%     directories = {'DUTOMRON'};
    directories = {'ECSSD','DUTOMRON','Pascal','HKUIS'};
    count = numel(directories);
    for i = 1:count
        filepath = fullfile(dirpath,directories{i},'Image');
        desdir = fullfile(dirpath,directories{i},[suffix,'_result']);
        if ~exist(desdir,'dir')
            mkdir(desdir);
        end
        generateSalMapList(filepath,desdir,net);
    end
    caffe.reset_all();
    
function generateSalMapList(dirpath,desDir,net)
    if ~exist(desDir,'dir')
        mkdir(desDir);
    end
    endings = {'.jpeg','.jpg','bmp','.png'};
    images = cellfun(@(x)dir([dirpath,filesep,'*',x]),endings,'UniformOutput',false);

    images = cat(1,images{:});
    count = size(images,1);

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