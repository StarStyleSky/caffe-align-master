function computeIoUsforImagelist()
   
    datasets = {'ECSSD','DUTOMRON','Pascal','HKUIS'};
    dirpath = '/media/yexiguafu/新加卷/Project02/SalObjectDetection';
    count = numel(datasets);
    for i = 1:count
        mat_file_path= fullfile(dirpath,datasets{i},'IoUs.mat');
        fprintf('%d/%d:%s\n',i,count,datasets{i});
        [ious] = computeIoUforImages(dirpath,datasets{i});
        save(mat_file_path,'ious','-v7.3');
    end
   
function [res] = computeIoUforImages(dirpath,filename)
    resname = 'last_result';
    
    gtDir = fullfile(dirpath,filename,'Ground');
    mapDir = fullfile(dirpath,filename,resname);
    endings = {'.png','.bmp','.jpg','.jpeg'};
    images = cellfun(@(x)dir([gtDir,filesep,'*',x]),endings,'UniformOutput',false);
    images = cat(1,images{:});
    count = size(images,1);
    ious = cell(count,1);
    for i = 1:count
        impath = fullfile(gtDir,images(i).name);
        [~,name,ext] = fileparts(images(i).name);
        esmFile = fullfile(mapDir,[name,ext]);
        gsm = imread(impath);
        esm = imread(esmFile);
        iou = computeIOU(esm,gsm);
        instance.filename = name;
        instance.iou = iou;
        ious{i} = instance;
    end
    res.dataname = filename;
    res.ious = ious;
    