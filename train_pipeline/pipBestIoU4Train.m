% BestIoU4Train

fprintf(1, 'Loading data list: '); tic
TRAIN_DATA_LIST = GetDataList( PARAM.Train_DataSet );
[gtBoxes, gtImIds, CategList, ~, gtDifficulties, DifficultyList] = bboxes_from_list( TRAIN_DATA_LIST );
imageN = length(TRAIN_DATA_LIST);
toc
fprintf(1, 'Loading proposed boxes: '); tic
boxes = GetProposedBoxes( PARAM.Train_DataSet );
toc

bestIoU   = cell(length(CategIdOfInterest), imageN);
bestGtIdx = cell(length(CategIdOfInterest), imageN);

for c = 1:length(CategIdOfInterest)
    fprintf( '%s : ', PARAM.CategOfInterest{c} );
    BEST_IoU_PATH = fullfile(SPECIFIC_DIRS.BestIoU4Train,sprintf('%s.mat',PARAM.CategOfInterest{c}));
    if exist(BEST_IoU_PATH,'file')
        fprintf(1,'Already Exists\n');
    else
        tic
        [~, gt_boxes_c]= reorganize_gt_wrt_image( ...
            imageN,CategIdOfInterest(c), gtImIds, gtBoxes );
        gt_boxes_c = vec(gt_boxes_c).';
        
        for k = 1:imageN
            [bestIoU{c,k},bestGtIdx{c,k}] = BestIoU_nonCell( boxes{k}, gt_boxes_c{k} );
        end
        
        MAT_CONTENT.bestIoU   = bestIoU(c,:);
        MAT_CONTENT.bestGtIdx = bestGtIdx(c,:);
        mkpdir_p( BEST_IoU_PATH );
        save( BEST_IoU_PATH, '-struct', 'MAT_CONTENT' );
        toc
    end
    clear MAT_CONTENT
end

