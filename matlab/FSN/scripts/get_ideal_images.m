function get_ideal_images() 
    dirpath = '/media/yexiguafu/新加卷/Project02/SalObjectDetection';
    datasets = {'iCoSeg','ECSSD','DUTOMRON','Pascal','HKUIS','iCoSeg'};
    dataset_name=  datasets{1};
    mat_file_path = fullfile(dirpath,dataset_name,'IoU.mat');
    load(mat_file_path);
    
    mat_file_path = fullfile(dirpath,dataset_name,'hole_ratio.mat');
    load(mat_file_path);
    
    %% 验证两个cell元组中的文件名是否对应
    [flags] = cellfun(@(x,y)check_names(x,y),ious,holeRatios,'UniformOutput',true);
    fprintf('not equal:%d\n',length(find(flags==0)));
    [fore_pickups] = cellfun(@(x,y)get_images_with_fore(x,y),ious,holeRatios,'UniformOutput',false);
    
    fore_pickups(cellfun(@(x)isempty(x),fore_pickups)) = [];
    disp(numel(fore_pickups));
    mat_file_path = fullfile(dirpath,dataset_name,'fore_pickup.mat');
    save(mat_file_path,'fore_pickups','-v7.3');
    
    [back_pickups] = cellfun(@(x,y)get_images_with_back(x,y),ious,holeRatios,'UniformOutput',false);
    back_pickups(cellfun(@(x)isempty(x),back_pickups)) = [];
    disp(numel(back_pickups));    
    mat_file_path = fullfile(dirpath,dataset_name,'back_pickup.mat');
    save(mat_file_path,'back_pickups','-v7.3');
    
    
    
function [res] =get_images_with_back(x,y)
    back_thresh = 0.7;
    hole_thresh = 0.4;
    res = [];
    if x.back_iou>back_thresh && y.holeratio > hole_thresh
        res.filename = x.filename;
        res.back_iou= x.back_iou;
        res.fore_iou = x.fore_iou;
        res.holeratio = y.holeratio;
    end
function [flag] = check_names(x,y)
    flag = strcmp(x.filename,y.name);
function [res]= get_images_with_fore(x,y)
    fore_thresh = 0.7;
    hole_thresh = 0.4;
    res = [];
    if x.fore_iou>fore_thresh && y.holeratio > hole_thresh
        res.filename = x.filename;
        res.back_iou= x.back_iou;
        res.fore_iou = x.fore_iou;
        res.holeratio = y.holeratio;
    end
    