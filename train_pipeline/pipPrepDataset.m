%% Start your code here

%% generate data list
fprintf( 1,'Generate data list : \n' );

for subset_idx = 1:length(PARAM.DataSubSet_Names)

    ds = PARAM.DataSubSet_Names{subset_idx};
    
    LIST_FILE_NAME = fullfile( SPECIFIC_DIRS.PrepDataset, 'list', [ds, '.mat'] );
    if exist( LIST_FILE_NAME, 'file' ) 
        fprintf( 1, 'Existed\n' );
        continue;
    end

    fprintf( ' - %s : ', ds ); tic
    
    % generate DATA_LIST

    list_type  = lower(ds);

    image_path      = fullfile( SPECIFIC_DIRS.VOC2007_ROOT, 'JPEGImages' );
    annotation_path = fullfile( SPECIFIC_DIRS.VOC2007_ROOT, 'Annotations' );
    set_path        = fullfile( SPECIFIC_DIRS.VOC2007_ROOT, 'ImageSets/Main' );

    set_path = fullfile(set_path,[list_type '.txt']);
    
    fid=fopen(set_path,'r');
    image_idx = textscan( fid, '%s' );
    fclose(fid);
    image_idx = image_idx{1};

    valid_ext = {'jpg','jpeg','JPG','JPEG','png','PNG', struct()};

    DATA_LIST = repmat( struct('im', '', 'obj', ...
        struct('type',{},'x1',{},'y1',{},'x2',{},'y2',{},...
        'difficulty',{},'truncation',{}) ), ...
        length(image_idx), 1 );

    for j = 1:length(image_idx)
        k = image_idx{j};
        annPath = fullfile(annotation_path,sprintf('%s.xml',k));
        if exist(annPath,'file')
            DATA_LIST(j)    = vocLoadXML( annPath );
            DATA_LIST(j).im = fullfile(image_path,DATA_LIST(j).im);
        else
            for r = 1:length(valid_ext)
                try
                    imPath = fullfile(image_path,sprintf('%s.%s',image_idx{j},valid_ext{r}));
                catch
                    error( 'Cannot find any image matched %s', image_idx{j} );
                end
                if exist(imPath,'file')
                    DATA_LIST(j).im = imPath;
                    break;
                end
            end        
        end
    end

    % get image dimension
    for k = 1:length( DATA_LIST )
        INFO = imfinfo( DATA_LIST(k).im );
        DATA_LIST(k).width  = INFO.Width;
        DATA_LIST(k).height = INFO.Height;
    end

    
    
    mkpdir_p( LIST_FILE_NAME );
    save( LIST_FILE_NAME, 'DATA_LIST', '-v7.3' );
    
    [~, ~, CategList] = bboxes_from_list( DATA_LIST );
    save( fullfile( SPECIFIC_DIRS.PrepDataset, [ds, '-categlist.mat'] ), 'CategList' );
    
    toc
    
end

%% generate dontcare class
DontCare_FILE_NAME = fullfile( SPECIFIC_DIRS.PrepDataset, 'dontcare.mat' );

fprintf(1,'Generate dont care list : ');

DontCareList = {'.*',''};
save( DontCare_FILE_NAME, 'DontCareList' );

fprintf(1,'Done\n');

