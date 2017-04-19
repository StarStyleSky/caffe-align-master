function demo()
    dirpath = '/media/yexiguafu/新加卷1/Project02/SalObjectDetection';
    datasets = {'iCoSeg','ECSSD','DUTOMRON','Pascal','HKUIS'};
    dataset_name=  datasets{5};
    mat_file_path = fullfile(dirpath,dataset_name,'IoU.mat');
    load(mat_file_path);
    
    filenames = cellfun(@(x)x.filename,ious,'UniformOutput',false);
    gaps = cellfun(@(x)get_gaps(x),ious,'UniformOutput',true);
    [~,idx] = sort(gaps,'descend');
    filenames = filenames(idx);
    topK = 100;
    count = numel(filenames);
    topK = max(topK,count);
    
    imgDir = fullfile(dirpath,dataset_name,'Image');
    foreDir = fullfile(dirpath,dataset_name,'foremap');
    backDir = fullfile(dirpath,dataset_name,'backmap');
    
    desDir = [imgDir,filesep,'../targets'];
    if ~exist(desDir,'dir')
        mkdir(desDir);
    end
    
    imageinfoes= cell(topK,1);
    for i = 1:topK
        filename = filenames{i};
        impath = fullfile(imgDir,[filename,'.png']);
        fore_map_file_path=  fullfile(foreDir,[filename,'.png']);
        back_map_file_path = fullfile(backDir,[filename,'.png']);
        
        foremap = imread(fore_map_file_path);
        backmap = imread(back_map_file_path);
        fusionmap = max(foremap,backmap);
        fusionmap = 255 - fusionmap;
        instance.name = filename;
        [ratios] = computeRegion(fusionmap);
        x = 10;
%         fusionmap = (foremap+backmap)/2;
        
%         imgpath= fullfile(desDir,[num2str(i),'_',filename,'.png']);
%         fore_file_path = fullfile(desDir,[num2str(i),'_',filename,'_fg.png']);
%         back_file_path = fullfile(desDir,[num2str(i),'_',filename,'_bg.png']);
%         fusion_file_path = fullfile(desDir,[num2str(i),'_',filename,'_fusion.png']);
%         copyfile(impath,imgpath);
%         copyfile(fore_map_file_path,fore_file_path);
%         copyfile(back_map_file_path,back_file_path);
%         imwrite(fusionmap,fusion_file_path);
        
    end
function [res] = get_gaps(x)
    res = abs(x.back_iou-x.fore_iou);