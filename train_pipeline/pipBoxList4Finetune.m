% Pipline stage: BoxList4Finetune

%% Start your code here

for subset_idx = 1:length(PARAM.DataSubSet_Names)

    ds = PARAM.DataSubSet_Names{subset_idx};

    fprintf(1, '================ %s\n', ds );
    
    BOX_LIST_FILE_PATH = fullfile( SPECIFIC_DIRS.BoxList4Finetune, [ds '.txt'] );
    if exist(BOX_LIST_FILE_PATH,'file');
        tfile = dir( BOX_LIST_FILE_PATH );
        if tfile.bytes>0
            fprintf('Existed\n');
            continue;
        end
    end
    
    fprintf( 1, 'Load data list : ' ); tic
    DATA_LIST = GetDataList(ds);
    imageN = length( DATA_LIST );
    [gtBoxes, gtImIds, CategList] = bboxes_from_list( DATA_LIST );
    toc
    
    fprintf( 1, 'Generate list for category ids : ' ); tic
    CATEGORY_LIST_PATH = fullfile( SPECIFIC_DIRS.BoxList4Finetune, 'category_list.txt' );
    if PARAM.KeepOriginalLabel
        [gt_categ_ids, gt_boxes] = ...
            reorganize_gt_wrt_image( imageN, CategIdOfInterest, gtImIds, gtBoxes );
        categ_id_list = CategIdOfInterest;
    else
        [gt_categ_ids, gt_boxes] = ...
            reorganize_gt_wrt_image( imageN, [], ...
            gtImIds(CategIdOfInterest), gtBoxes(CategIdOfInterest) );
        categ_id_list = 1:length(CategIdOfInterest);
    end
    fid = fopen( CATEGORY_LIST_PATH, 'w' );
    for c = 1:length(CategIdOfInterest)
        fprintf( fid, '%d %s\n', categ_id_list(c), PARAM.CategOfInterest{c} );
    end
    fclose(fid);
    toc
    
    fprintf( 1, 'Get boxes : ' ); tic
    boxes = GetProposedBoxes( ds );
    toc
    
    fprintf( 1, 'Generating list : \n' );
    
    window_file = sprintf( BOX_LIST_FILE_PATH );
    fid = fopen(window_file, 'wt');

    channels = 3; % three channel images

    for k = 1:imageN
      tic_toc_print('make window file: %d/%d\n', k, imageN);
      img_path = DATA_LIST(k).im;
      
      num_boxes = size(boxes{k}, 1);
      
      [bestIoU, bestGtIdx] = BestIoU_nonCell( boxes{k}, gt_boxes{k} );
      label = zeros( length(bestIoU), 1 );
      label( bestGtIdx>0 ) = gt_categ_ids{k}(bestGtIdx(bestGtIdx>0));
      label( bestIoU < 1e-5 ) = 0;
      bestIoU( bestIoU < 1e-5 ) = 0;
      
      outputIoU   = bestIoU;
      label( outputIoU<1e-5 ) = 0;
      outputIoU( outputIoU<1e-5 ) = 0;
      
      with_gt_idxb = label>0;
      associate_gt_boxes = nan( num_boxes, 4 );
      associate_gt_boxes( with_gt_idxb, : ) = ...
          gt_boxes{k}(bestGtIdx( with_gt_idxb ),:);
      
      fprintf(fid, '# %d\n', k-1);
      fprintf(fid, '%s\n', img_path);
      fprintf(fid, '%d\n%d\n%d\n', ...
          channels, ...
          DATA_LIST(k).height, ...
          DATA_LIST(k).width );
      fprintf(fid, '%d\n', num_boxes);
      for j = 1:num_boxes
        bbox = boxes{k}(j,:)-1; % 1-base to 0-base
        %if with_gt_idxb(j)
        %    agt_bbox = associate_gt_boxes(j,:) - 1;
        %    fprintf(fid, '%d %.3f %d %d %d %d %d %d %d %d\n', ...
        %        label(j), outputIoU(j), bbox(2), bbox(1), bbox(4), bbox(3), ...
        %        agt_bbox(2), agt_bbox(1), agt_bbox(4), agt_bbox(3) );
        %else
            fprintf(fid, '%d %.3f %d %d %d %d\n', ...
                label(j), outputIoU(j), bbox(2), bbox(1), bbox(4), bbox(3));
        %end
      end
    end

    fclose(fid);
    
end


