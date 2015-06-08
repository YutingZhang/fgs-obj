function [Boxes, ImIds, CategList, BoxIdsInImage, Difficulty, DifficultyList ] = ...
    bboxes_from_list( DATA_LIST, CategList0 )
% [Boxes, ImIds, CategList, BoxIdsInImage, Difficulty, DifficultyList] = ...
%        kittiBBoxFromList( DATA_LIST, 'default' )
% Boxes -- Organize all the bounding boxes according to categories. 
%  For each category, bboxes are organized in a 4xN matrix. [y1 x1 y2 x2]
% ImIds -- the image ids of the bboxes
% CategList -- string cell for category name 

% extract and combine obj
OBJ = { DATA_LIST.obj }.';
NumObj4Img = cellfun( @length, OBJ );
OBJ   = cat(2,OBJ{:}).';

% % compute difficulties
% if nargout>=5
%     % ******** need updates for non-kitti dataset
%     % bascally, this should by done by *GetList instead of this function
%     di = kittiDifficulties( DATA_LIST );
%     di = [ di{:} ];
% else
%     di = zeros( size(OBJ) );
% end

% generate overall output
[CategList,~,ClassID]= unique( { OBJ.type } );
if exist('CategList0','var') && ~isempty(CategList0)
    CategList = [ CategList0, setdiff( CategList, CategList0 ) ];
    [~,ClassID] = ismember( { OBJ.type }, CategList );
end

BOXES = [ [OBJ.y1] ; [OBJ.x1]; [OBJ.y2]; [OBJ.x2] ].';

ImIDX = arrayfun( @(id,n) {repmat(id,n,1)}, (1:length(NumObj4Img)).', NumObj4Img );
ImIDX = cell2mat( ImIDX );

BoxIDX = arrayfun( @(n) {(1:n).'}, NumObj4Img );
BoxIDX = cell2mat( BoxIDX );

if isfield(OBJ,'difficulty')
    DIFFICULTY = [OBJ.difficulty].';
else
    DIFFICULTY = zeros( numel(OBJ), 1 );
end

% partition result according to categories

ImIds = cell(length(CategList),1);
Boxes = cell(length(CategList),1);
BoxIdsInImage   = cell(length(CategList),1);
Difficulty = cell(length(CategList),1);

for k = 1:length(CategList)
    chosenIdx = ( ClassID == k );
    Boxes{k} = BOXES( chosenIdx, : );
    ImIds{k} = ImIDX(chosenIdx);
    BoxIdsInImage{k}   = BoxIDX(chosenIdx);
    Difficulty{k} = DIFFICULTY(chosenIdx);
end

DifficultyList = unique( DIFFICULTY );

end
