function get_target_image03()
    dirpath = '/media/yexiguafu/新加卷1/Project02/SalObjectDetection/DUTOMRON';
    imgDir = fullfile(dirpath,'Image');
    foreDir=  fullfile(dirpath,'foremap');
    backDir = fullfile(dirpath,'backmap');
    
    endings = {'.jpg','.png','.bmp','.jpeg'};
    images = cellfun(@(x)dir([imgDir,filesep,'*',x]),endings,'UniformOutput',false);
    images = cat(1,images{:});
    count = size(images,1);
    for i = 1:count
        [~,name,~] = fileparts(images(i).name);
        fore_map_file_path = fullfile(foreDir,[name,'.png']);
        back_map_file_path = fullfile(backDir,[name,'.png']);
        foremap = imread(fore_map_file_path);
        backmap = imread(back_map_file_path);
        fusion = max(foremap,backmap);
        [ratios] = computeRegion(fusion);
        if numel(ratios)>1
            x = 10;
        end
        
    end
    