function detInitPath
% DETINITPATH initializes the dependency paths for single image detection
%

TOOLBOX_ROOT_DIR = fileparts(which(mfilename('fullpath')));
addpath( genpath( fullfile( TOOLBOX_ROOT_DIR, 'dependency' ) ) );
addpath( genpath( fullfile( TOOLBOX_ROOT_DIR, 'caffe/matlab/caffe' ) ) );
addpath( TOOLBOX_ROOT_DIR );

