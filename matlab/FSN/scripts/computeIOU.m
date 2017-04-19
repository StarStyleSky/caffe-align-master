function [iou] = computeIOU(esm,gtmap)
    esm = single(esm);
    gtmap = single(gtmap);
    
    [h,w,~] = size(esm);
    
    if size(gtmap,1) ~= h || size(gtmap,2) ~= w
        esm = mat2gray(esm,[size(gtmap,1),size(gtmap,2)],'bilinear');
    end
   
    if max(esm(:))>1
        esm = esm/255;
    end
    if max(gtmap(:))>1
        gtmap = gtmap/255;
    end
    overlay = esm.*gtmap;
    union = esm+gtmap-overlay;
    iou = nnz(overlay)/nnz(union);