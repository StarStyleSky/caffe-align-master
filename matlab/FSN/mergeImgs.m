function mergeImgs()
    dirpath = '/media/yexiguafu/新加卷1/Project02/SalObjectDetection';
    desDir = '/media/yexiguafu/新加卷1/Project02/SalObjectDetection/AllData';
    if ~exist(desDir,'dir')
        mkdir(desDir);
    end
    files = {'DUTOMRON','ECSSD','HKUIS','Pascal','MSRA10K'};

    count = numel(files);
    for i = 1:count
            copyImg(dirpath,files{i},desDir);

    end
function copyImg(dirpath,filename,desDir)

    endings = {'.jpg','.jpeg','.bmp','.png'};
    images = cellfun(@(x)dir([fullfile(dirpath,filename,'Image'),filesep,'*',x]),endings,'UniformOutput',false);
    images = cat(1,images{:});
    
    count = numel(images);
    for i = 1:count
        impath = fullfile(fullfile(dirpath,filename,'Image'),images(i).name);
        fprintf('%d/%d:%s.\n',i,count,impath);
        [~,name,ext] = fileparts(impath);
        filepath = fullfile(desDir,[filename,'_',name,ext]);
        if ~exist(filepath,'file')
            copyfile(impath,filepath);
        end
    end