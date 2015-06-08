function trainCallStage( STAGE_NAME, PARAM, SPECIFIC_DIRS, CategIdOfInterest )
% TRAINCALLSTAGE calls a given training stage with user-specified parameters
% 
% Intro:
%   For a training stage with the name STAGE_NAME, its code is at 
%       ./train_pipeline/pip(STAGE_NAME).m
%   While it is OK to call this script directly, calling it through
%     trainCallStage can
%   1) handle the parameters in a cleaner way;
%   2) avoid intermediate variable mixed in the MATLAB base work space;
%   3) keep track whether a stage is complete or not
%
% Usage:
%
%   trainCallStage( STAGE_NAME, PARAM )
%
% Input:
%
%   STAGE_NAME: should be a string indicating the name of the stage
%    List of stages:
%          BBoxRegTrain
%          BestIoU4Train
%          BoxList4Finetune
%          FeatureNorm4Train
%          FeatureNorm4Train_Internal
%          Features4GPTrain
%          Features4Groundtruth
%          Features4Proposed
%          GPTrain
%          PrepDataset
%          RegionProposal
%          RegionProposal4GPTrain
%          Test4GPTrain
%          Train
%
%   PARAM: should be a struct consisting of the full parameter set for training
%       Refer to trainInit for more details.
%    If PARAM is not specified or empty, the variable value of
%       PARAM in the caller will be used.
%
%   SPECIFIC_DIRS: should be a vector of integer number that indicates
%     which categories should be included in training
%       Refer to trainInit for more details.
%    If SPECIFIC_DIRS is not specified or empty, the variable value of
%       SPECIFIC_DIRS in the caller will be used.
%
%   CategIdOfInterest: should be a struct consisting of the cache folder paths
%     for all the stages
%       Refer to trainInit for more details.
%    If CategIdOfInterest is not specified or empty, the variable value of
%       CategIdOfInterest in the caller will be used.
%

if ~exist( 'PARAM', 'var' ) || isempty( PARAM )
    PARAM = evalin( 'caller', 'PARAM' );
end
if ~exist( 'SPECIFIC_DIRS', 'var' ) || isempty( SPECIFIC_DIRS )
    SPECIFIC_DIRS = evalin( 'caller', 'SPECIFIC_DIRS' );
end
if ~exist( 'CategIdOfInterest', 'var' ) || isempty( CategIdOfInterest )
    CategIdOfInterest = evalin( 'caller', 'CategIdOfInterest' );
end

STAGE_SPECIFIC_DIR = eval( sprintf('SPECIFIC_DIRS.%s', STAGE_NAME) );
STAGE_COMPLETE_FILENAME = fullfile(STAGE_SPECIFIC_DIR, 'COMPLETE');

fprintf( '****** Call stage : %s : ', STAGE_NAME );

if exist( STAGE_COMPLETE_FILENAME, 'file' )
    fprintf( 'Complete already\n' );
    return;
end
fprintf( 'Run\n' );

eval( sprintf( 'pip%s', STAGE_NAME ) );

fclose( fopen( STAGE_COMPLETE_FILENAME, 'w' ) );

end
