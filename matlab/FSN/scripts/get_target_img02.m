function get_target_img02()
    dirpath = '/media/yexiguafu/新加卷/Project02/SalObjectDetection';    
    
%     datasets = {'ECSSD','DUTOMRON','Pascal','HKUIS'};
    datasets = {'iCoSeg'};
    foreName = 'foremap';
    backName = 'backmap';
    count = numel(datasets);
   
    for i =1:count
       
        gtDir = fullfile(dirpath,datasets{i},'Ground');
        foreDir = fullfile(dirpath,datasets{i},foreName);
        backDir = fullfile(dirpath,datasets{i},backName);
        get_best(foreDir,backDir,gtDir);
    end
function get_best(foreDir,backDir,gtDir)
%     desDir = fullfile(gtDir,'../pickups');
%     holeDir = fullfile(gtDir,'../holes');
%     if exist(desDir,'dir')
%         unix(['rm -rf ',desDir]);
%        
%     end
%     mkdir(desDir);
%     
%     if exist(holeDir,'dir')
%         unix(['rm -rf ',holeDir]);
%         
%     end
%     mkdir(holeDir);
    
    endings = {'.jpeg','.jpg','bmp','.png'};
    images = cellfun(@(x)dir([gtDir,filesep,'*',x]),endings,'UniformOutput',false);

    images = cat(1,images{:});
    count = size(images,1);
    
    ious = cell(count,1);

    for i = 1:count
        
        gtfile = fullfile(gtDir,images(i).name);
        fprintf('%d/%d:%s.\n',i,count,gtfile);
        [~,name,~] = fileparts(gtfile);
        backfile = fullfile(backDir,[name,'.png']);
        forefile = fullfile(foreDir,[name,'.png']);
        
        gtmap = imread(gtfile);
        foremap = imread(forefile);
        backmap = imread(backfile);

        back_iou = computeIOUforBack(backmap,255-gtmap);
        fore_iou = computeIOU(foremap,gtmap);
        
        instance.filename = name;
        instance.back_iou = back_iou;
        instance.fore_iou = fore_iou;
        ious{i} = instance;
        
    end
    mat_file_path = [gtDir,filesep,'../IoU.mat'];
    save(mat_file_path,'ious','-v7.3');
    
%     back_scores = cellfun(@(x)x.back_iou,diff,'UniformOutput',true);
%     fore_scores = cellfun(@(x)x.fore_iou,diff,'UniformOutput',true);    
%     back_thresh = 0.8;
%     fore_thresh = 0.8;
%     count = numel(diff);
%     for i = 1:count
%         
%         if diff{i}.back_iou > back_thresh && diff{i}.fore_iou > fore_thresh
%             filepath = fullfile(desDir,[diff{i}.filename,'.png']);
%             impath = fullfile(gtDir,[diff{i}.filename,'.png']);
%             copyfile(impath,filepath);
% 
%             filepath = fullfile(desDir,[diff{i}.filename,'_fg.png']);
%             impath = fullfile(foreDir,[diff{i}.filename,'.png']);
%             copyfile(impath,filepath);
%             foremap = imread(impath);
% 
%             filepath = fullfile(desDir,[diff{i}.filename,'_bg.png']);
%             impath = fullfile(backDir,[diff{i}.filename,'.png']);
%             copyfile(impath,filepath);
%             backmap = imread(impath);
%             
%             fusion_file_path = fullfile(holeDir,[diff{i}.filename,'_fusion.png']);
%             fusion = max(foremap,backmap);
%             
%             imwrite(fusion,fusion_file_path);
%         end
%     end
    