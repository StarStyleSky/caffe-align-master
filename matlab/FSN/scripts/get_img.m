function get_img()
   dirpath = '/media/yexiguafu/新加卷/Project02/SalObjectDetection';    
%    datasets = {'ECSSD','DUTOMRON','Pascal','HKUIS'};
   datasets = {'iCoSeg'};
   count = numel(datasets);
   thresh_name = 'thresh0.7_hole0.4';
   for i = 1:count
       dataDir = fullfile(dirpath,datasets{i});
       get_images(dataDir,thresh_name);
   end
function get_images(dirpath,name)
   
    mstrs = regexp(dirpath,filesep,'split');
    dirname = mstrs{end};
    mat_file_path= fullfile(dirpath,'back_pickup.mat');
    load(mat_file_path);
    
    desDir = [dirpath,filesep,name,filesep,'ImageBacks'];
    
    imgDir = fullfile(dirpath,'Image');
    foreDir = fullfile(dirpath,'foremap');
    backDir = fullfile(dirpath,'backmap');
    
    if ~exist(desDir,'dir')
        mkdir(desDir);
    end
    
    count = size(back_pickups,1);
    for i =1:count
        imginfo = back_pickups{i};
        name = imginfo.filename;
        
        if strcmp(dirname,'HKUIS')
            impath = fullfile(imgDir,[name,'.png']);
            img_path = fullfile(desDir,[name,'.png']);
        else
            impath = fullfile(imgDir,[name,'.jpg']);
            img_path = fullfile(desDir,[name,'.jpg']);
        end
        foremap = fullfile(foreDir,[name,'.png']);
        backmap = fullfile(backDir,[name,'.png']);
        
        fore_map_path = fullfile(desDir,[name,'_fg.png']);
        back_map_path = fullfile(desDir,[name,'_bg.png']);
        
        fore = imread(foremap);
        back = imread(backmap);
        fusion = max(fore,back);
        fusion_path = fullfile(desDir,[name,'_fusion.png']);
        copyfile(impath,img_path);
        copyfile(foremap,fore_map_path);
        copyfile(backmap,back_map_path);
        imwrite(fusion,fusion_path);
    end