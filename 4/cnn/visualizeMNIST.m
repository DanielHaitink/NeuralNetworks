addpath ../common/;
images = loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);


tiles = num2cell(images(:,:,1:400),[1 2]);
tiles = reshape(tiles,20,20);
out = cell2mat(tiles);

imshow(1 - out)
        