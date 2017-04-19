function get_target_img01()
    dirpath = '/media/yexiguafu/新加卷/Project02/SalObjectDetection';    
    
    datasets = {'ECSSD','DUTOMRON','HKUIS','Pascal'};
    foreName = 'foremap';
    backName = 'backmap';
    count = numel(datasets);
   
    for i =3:3
       
        gtDir = fullfile(dirpath,datasets{i},'Ground');
        foreDir = fullfile(dirpath,datasets{i},foreName);
        backDir = fullfile(dirpath,datasets{i},backName);
        get_best(foreDir,backDir,gtDir);
    end
function get_best(foreDir,backDir,gtDir)
%     desDir = fullfile(gtDir,'../pickups01');
%     
%     if exist(desDir,'dir')
%         unix(['rm -rf ',desDir]);
%        
%     end
%     mkdir(desDir);
    
    endings = {'.jpeg','.jpg','bmp','.png'};
    images = cellfun(@(x)dir([gtDir,filesep,'*',x]),endings,'UniformOutput',false);

    images = cat(1,images{:});
    count = size(images,1);
    
    diff = cell(count,1);
    for i = 1:count
        
        gtfile = fullfile(gtDir,images(i).name);
        fprintf('%d/%d:%s.\n',i,count,gtfile);
        [~,name,~] = fileparts(gtfile);
        backfile = fullfile(backDir,[name,'.png']);
        forefile = fullfile(foreDir,[name,'.png']);

        foremap = imread(forefile);
        backmap = imread(backfile);

        contramap = 255 - backmap;
        iou = computeIoU(foremap,contramap);
        
        instance.filename = name;
        instance.score = iou;
        diff{i} = instance;
        
    end
    x = 10;
%     scores = cellfun(@(x)x.score,diff,'UniformOutput',true);
%     [~,idx] = sort(scores,'ascend');
%     
%     diff = diff(idx);
%     iou_thresh = 0.75;
%     count = numel(scores);
%     for i = 1:count
%         if diff{i}.score < iou_thresh
%             filepath = fullfile(desDir,[diff{i}.filename,'.png']);
%             impath = fullfile(gtDir,[diff{i}.filename,'.png']);
%             copyfile(impath,filepath);
% 
%             filepath = fullfile(desDir,[diff{i}.filename,'_fg.png']);
%             impath = fullfile(foreDir,[diff{i}.filename,'.png']);
%             copyfile(impath,filepath);
% 
% 
%             filepath = fullfile(desDir,[diff{i}.filename,'_bg.png']);
%             impath = fullfile(backDir,[diff{i}.filename,'.png']);
%             copyfile(impath,filepath);
%         end
%     end
%     