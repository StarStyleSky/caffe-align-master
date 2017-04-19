function get_image_list()
    dirpath = '/media/yexiguafu/新加卷/Project02/SalObjectDetection';
    datasets = {'iCoSeg','HKUIS','ECSSD','Pascal','DUTOMRON'};
    count = numel(datasets);
    for i = 1:count
        foreDir = fullfile(dirpath,datasets{i},'foremap');
        backDir = fullfile(dirpath,datasets{i},'backmap');
        gsmDir= fullfile(dirpath,datasets{i},'Ground');
        
        [backIoU] = computeIoUsbackGround(backDir,gsmDir);
        [foreIoU] = computeIoUsforeGround(foreDir,gsmDir);
        
        mat_file_path = fullfile(dirpath,datasets{i},[datasets{i},'_iou.mat']);
        save(mat_file_path,'ious','-v7.3');
    end
function [ious] =computeIoUsbackGround(dirpath,gsmDir)
    endings = {'.jpg','.png','.bmp','.jpeg'};
    images = cellfun(@(x)dir([dirpath,filesep,'*',x]),endings,'UniformOutput',false);
    images = cat(1,images{:});
    count = size(images,1);
    ious = cell(count,1);
    for i = 1:count
        impath = fullfile(dirpath,images(i).name);
        [~,name,~] = fileparts(impath);
        fprintf('%d/%d:%s\n',i,count,impath);
        gsmpath = fullfile(gsmDir,[name,'.png']);
        esm = imread(impath);
        gsm = imread(gsmpath);
        if max(gsm(:)) ==1
            gsm = uint8(255*gsm);
        end
            
        ious{i} = computeIOUbackGround(esm,255-gsm);
    end
function [ious] = computeIoUsforeGround(dirpath,gsmDir)
    endings = {'.jpg','.png','.bmp','.jpeg'};
    images = cellfun(@(x)dir([dirpath,filesep,'*',x]),endings,'UniformOutput',false);
    images = cat(1,images{:});
    count = size(images,1);
    ious = cell(count,1);
    for i = 1:count
        impath = fullfile(dirpath,images(i).name);
        [~,name,~] = fileparts(impath);
        fprintf('%d/%d:%s\n',i,count,impath);
        gsmpath = fullfile(gsmDir,[name,'.png']);
        esm = imread(impath);
        gsm = imread(gsmpath);
        if max(gsm(:)) ==1
            gsm = uint8(255*gsm);
        end
        ious{i} = computeIOUforeGround(esm,gsm);
    end
