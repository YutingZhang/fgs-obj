#!/bin/bash

echo Get Selective Search Toolbox

wget -c http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip
unzip SelectiveSearchCodeIJCV.zip
rm -f SelectiveSearchCodeIJCV.zip
mkdir -p dependency/SelectiveSearch
mv SelectiveSearchCodeIJCV/* dependency/SelectiveSearch/
rmdir SelectiveSearchCodeIJCV

echo Get minFunc toolbox
wget -c http://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip
unzip minFunc_2012.zip
rm -f minFunc_2012.zip
rm `find minFunc_2012 -name '*.mex*'`
mkdir -p dependency/minFunc
mv minFunc_2012/* dependency/minFunc/
rmdir minFunc_2012

echo Get GPML toolbox

wget http://gaussianprocess.org/gpml/code/matlab/release/gpml-matlab-v3.4-2013-11-11.tar.gz
tar -xzf gpml-matlab-v3.4-2013-11-11.tar.gz
rm -f gpml-matlab-v3.4-2013-11-11.tar.gz
mv gpml-matlab-v3.4-2013-11-11 dependency/


