function S3_2_Eval(dirpath,gtDir)


    %% salient map filepath
    salPath = dirpath;
    
    %% GroundTruth filepath
    gtPath = gtDir;
    dir_gt= dir([gtPath filesep '*.png']);
    imNum = length(dir_gt);
%     imNum = 5;
    precision = zeros(256,1);
    recall = zeros(256,1);
    MAE = 0;
    p = 0;
    r = 0;
    %% compute pr curve
    disp('Compute Evaluation:');
    for i = 1:imNum
        imName = dir_gt(i).name;
        [~,name,~] = fileparts(imName);
        fprintf('\t processing %d (%s)\n',i,name);
        input_im = imread([salPath '/' name '.png']);
        truth_im = imread([gtPath '/' name '.png']);
        if max(truth_im(:))<128
            truth_im = 255*truth_im;
        end
        truth_im = double(truth_im>128); 
        truth_im = truth_im(:,:,1); % 0-1

        input_im = imresize(input_im,[size(truth_im,1),size(truth_im,2)]);

        % 0-255 cut; input_im should be 0-255
        for threshold = 0:255
            index1 = (input_im>=threshold);
            truePositive = length(find(index1 & truth_im));
            groundTruth = length(find(truth_im));
            detected = length(find(index1));
            if truePositive~=0
                precision(threshold+1) = precision(threshold+1)+truePositive/detected;
                recall(threshold+1) = recall(threshold+1)+truePositive/groundTruth;
            end
        end 

        % mae
        input_im = double(input_im(:,:,1))./255; % 0-1
        MAE = MAE + mean2(abs(truth_im-input_im));

        % 2*means
        threshold = 2*mean2(input_im);
        sal_cut_im = double(input_im >=threshold); % 0-1
        truePositive = length(find(sal_cut_im & truth_im));
        groundTruth = length(find(truth_im));
        detected = length(find(sal_cut_im));
        if truePositive~=0
            p = p+truePositive/detected;
            r = r+truePositive/groundTruth;
        end


    end
    precision = precision./imNum;
    recall = recall./imNum;
    MaxFmeasure = max(1.3*precision.*recall./(0.3*precision+recall));
    MeanFmeasure = mean(1.3*precision.*recall./(0.3*precision+recall));

    adapprecision = p/imNum;
    adaprecall = r/imNum;
    AdapFmeasure = 1.3*adapprecision*adaprecall/(0.3*adapprecision+adaprecall);

    MAE = MAE/imNum;
    fprintf('Max Fmeasure:%f;\n Mean Fmeasure:%f;\n Adaptive Fmeasure:%f;\n MAE:%f\n',MaxFmeasure,MeanFmeasure,AdapFmeasure,MAE);
