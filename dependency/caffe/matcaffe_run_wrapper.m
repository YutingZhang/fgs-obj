function [F, sizeSingleF] = matcaffe_run_wrapper( patches, image_means, response_ids_or_names, output_type )
% [F, sizeSingleF] = matcaffe_run_wrapper( patches, image_means, response_ids_or_names, output_type )
%  return features in a 1x1 cell, for the compatiblity with DeCAF
%  when image_means are larger than patches, its center part will be
%    cropped using CAFFE's protocol
% output_type : 'vec-c' (default, row-column order), 'vec-matlab' (column-row order), 
%               'array-c', 'array-matlab'

if ~exist( 'output_type', 'var' ) || isempty(output_type)
    output_type = 'vec-c';
end

if ~ismember( output_type, {'vec-c','vec-matlab','array-c','array-matlab'} )
    error( 'Unrecognized output_type' );
end


global caffe_instance_salt
global caffe_response_info
global caffe_output_response_ids

cur_caffe_instance_salt = caffe( 'get_init_key' );
if isempty(cur_caffe_instance_salt) || cur_caffe_instance_salt ~= caffe_instance_salt
    error( 'Please initialize caffe with matcaffe_init' );
end

if ~exist( 'response_ids_or_names', 'var' ) || isempty(response_ids_or_names)
    response_ids = caffe_output_response_ids.';
else
    response_ids_or_names = reshape(response_ids_or_names,1,numel(response_ids_or_names));
    if ischar( response_ids_or_names )
        response_ids_or_names = { response_ids_or_names };
    end
    if iscell( response_ids_or_names )
        response_ids = zeros(1,length(response_ids_or_names));
        nameIdxB = cellfun( @ischar, response_ids_or_names );
        [~,response_ids(nameIdxB)] = ismember( response_ids_or_names(nameIdxB), {caffe_response_info.name} );
        response_ids(~nameIdxB) = cellfun( @double, response_ids_or_names(~nameIdxB));
    else
        response_ids = double( response_ids_or_names );
    end
end

if any( response_ids<1 ) || any( response_ids>length(caffe_response_info) )
    error( 'response_ids out of range' );
end

patches     = single( patches );

patchSize = [ size(patches,1), size(patches,2) ];

if exist( 'image_means', 'var' ) && ~isempty(image_means)
    image_means = single( image_means );
    if isscalar(image_means)
        patches = patches - image_means;
    else
        patches = bsxfun( @minus, patches, image_center_patch( image_means, patchSize ) );
    end
end
patches(isnan(patches(:))) = single(0); % handle padding

F = matcaffe_run( patches, response_ids );

if ismember( output_type, {'array-c','vec-matlab','array-matlab'} )
    for i = length(F)
        F{i} = reshape( F{i}, [caffe_response_info(response_ids(i)).size(1:3) size(F{i},2)] );
    end
    if ismember( output_type, {'vec-matlab','array-matlab'} )
        for i = length(F)
            F{i} = permute( F{i}, [2 1 3 4] );
        end
        if ismember( output_type, {'vec-matlab'} )
            for i = length(F)
                Sz_i = size(F{i});
                F{i} = reshape( F{i}, [prod(Sz_i(1:3)),Sz_i(4)] );
            end
        end
    end   
end

if nargout>=2
    sizeSingleF = cat( 1, caffe_response_info(response_ids).size );
    sizeSingleF = sizeSingleF(:,1:3);
    if ismember( output_type, {'vec-matlab','array-matlab'} )
        sizeSingleF = sizeSingleF(:,[2 1 3]);
    end
end

end
