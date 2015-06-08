function codc = GetDontCare( coi, PrepDataset_SpecificDir )

if ( ~exist( 'PrepDataset_SpecificDir', 'var' ) || isempty(PrepDataset_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    PrepDataset_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.PrepDataset' );
end

if iscell(coi)
    error( 'COI cannot be cell.' );
end

DontCare_FILE_NAME = fullfile( PrepDataset_SpecificDir, 'dontcare.mat' );
MAT_CONTENT = load( DontCare_FILE_NAME );
DontCareList = MAT_CONTENT.DontCareList;

if isempty(coi)
    coi = ' ';
end

codc = {};
for k = 1:size(DontCareList,1)
    if ~isempty( regexp( coi, [ '^' DontCareList{k,1} '$' ], 'once' ) )
        dl = DontCareList{k,2};
        if isempty(dl)
        elseif iscell(dl)
            codc = [codc,reshape(dl,1,numel(dl))];
        elseif ischar(dl)
            codc = [codc,{dl}];
        else
            error( 'Unrecognized DontCare info' );
        end
    end
end
codc = unique( codc );

end
