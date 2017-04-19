function get_holes()
    dirpath = '/media/yexiguafu/新加卷1/Project02/SalObjectDetection';    
    
%     datasets = {'DUTOMRON','HKUIS','Pascal'};
    datasets = {'iCoSeg'};
    foreName = 'foremap';
    backName = 'backmap';
    count = numel(datasets);
   
    for i = 1:count      
        gtDir = fullfile(dirpath,datasets{i},'Ground');
        foreDir = fullfile(dirpath,datasets{i},foreName);
        backDir = fullfile(dirpath,datasets{i},backName);
        holeRatios = get_hole_informations(foreDir,backDir,gtDir);
        mat_file_path= fullfile(foreDir,'../hole_ratio.mat');
        save(mat_file_path,'holeRatios','-v7.3');
    end
function [imageinfoes] = get_hole_informations(foreDir,backDir,gtDir)
    endings = {'.jpeg','.jpg','.png','.bmp'};
    images = cellfun(@(x)dir([gtDir,filesep,'*',x]),endings,'UniformOutput',false);
    images = cat(1,images{:});
    count = size(images,1);
    
    imageinfoes = cell(count,1);
    for i = 1:count
        impath = fullfile(gtDir,images(i).name);
        [~,name,ext] = fileparts(impath);
        fprintf('%d/%d:%s.\n',i,count,[name,ext]);
        fore_map_file_path = fullfile(foreDir,[name,ext]);
        back_map_file_path = fullfile(backDir,[name,ext]);
        foremap = imread(fore_map_file_path);
        backmap = imread(back_map_file_path);
        fusionmap = max(foremap,backmap);
        hole_ratio  = computeRegion(fusionmap);
        instance.name = name;
        instance.holeratio = hole_ratio;
        imageinfoes{i} = instance;
    end