function [categIds, varargout] = reorganize_gt_wrt_image( imageN, CategIdOfInterest, gtImIds, varargin )

if ~iscell(gtImIds)
    gtImIds = {gtImIds};
    INPUT = cellfun( @(a) {{a}}, varargin );
else
    INPUT = varargin;
end

if isempty( CategIdOfInterest )
    CategIdOfInterest = 1:length(gtImIds);
end

categIds0 = arrayfun( @(a,b) {repmat( b, length(a{1}),1 )} , ...
        gtImIds(CategIdOfInterest), CategIdOfInterest.' );
categIds0 = cell2mat( categIds0 );
imIds0 = cell2mat( vec(gtImIds(CategIdOfInterest)) );

if isempty(imageN)
    imageN = max(imIds0);
end

M = length( INPUT );

V0 = cell(1,M);
varargout = repmat({cell(imageN,1)},1,M+1);

for i=1:M
    V0{i} = cell2mat( vec(INPUT{i}(CategIdOfInterest)) );
end


categIds = cell( imageN,1 );

for k = 1:imageN
    chosenIdxB = (imIds0 == k);
    categIds{k} = categIds0(chosenIdxB);
    for i=1:M
        varargout{i}{k} = V0{i}(chosenIdxB,:);
    end
end

numTotalBoxes  = length( imIds0 );
varargout{M+1} = numTotalBoxes;

end
