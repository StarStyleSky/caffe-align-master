function get_target_img(dirpath,name)
    mstrs = regexp(dirpath,filesep,'split');
    dirname = mstrs{end};
    mat_file_path= fullfile(dirpath,'fore_pickup.mat');
    load(mat_file_path);
    
    desDir = [dirpath,filesep,name,filesep,'ImageFores'];
    
    imgDir = fullfile(dirpath,'Image');
    foreDir = fullfile(dirpath,'foremap');
    backDir = fullfile(dirpath,'backmap');
    
    if ~exist(desDir,'dir')
        mkdir(desDir);
    end
    
    count = size(fore_pickups,1);
    for i =1:count
        imginfo = fore_pickups{i};
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