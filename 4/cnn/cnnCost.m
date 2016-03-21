function [cost, grad, preds, activations] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations

%%% REPLACE THE FOLLOWNG LINE %%%
activations =  cnnConvolve(filterDim, numFilters, images, Wc, bc);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations

%%% REPLACE THE FOLLOWNG LINE %%%
activationsPooled = cnnPool(poolDim, activations);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
%%% REPLACE THE FOLLOWING LINE %%%
activationsPooled = reshape(activationsPooled,[2000,256]);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.

%%% COMPUTE THE SOFTMAX OUTPUT %%%
probs = zeros(numClasses,numImages);
Y_wx=Wd*activationsPooled;
Y_wxb = bsxfun(@plus, Y_wx, bd);
Y_num(:,:) = exp(Y_wxb(:,:));
probs = bsxfun(@rdivide,Y_num,sum(Y_num,1));
%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.
indexes = sub2ind(size(probs), labels', 1:numImages);
cost = -mean(log(probs(indexes)));

% Makes predictions given probs and returns without backproagating errors.
[~,preds] = max(probs,[],1);
preds = preds';
if pred
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

deriv = probs;
deriv(indexes) = deriv(indexes) - 1;
deriv = deriv ./ numImages;

Wd_grad = deriv * activationsPooled';
bd_grad = sum(deriv, 2);

deriv2_pooled = Wd' * deriv;
deriv2_pooled = reshape(deriv2_pooled, outputDim, outputDim, numFilters, numImages);
delta_upsampled = zeros(convDim, convDim, numFilters, numImages);

for im_idx=1:numImages
    im = squeeze(images(:,:,im_idx));
    for f_idx=1:numFilters
        delta_pool = (1/poolDim^2) * kron(squeeze(deriv2_pooled(:,:,f_idx,im_idx)), ones(poolDim));
        delta_upsampled(:,:,f_idx, im_idx) = delta_pool .* ...
            activations(:,:,f_idx,im_idx).*(1-activations(:,:,f_idx,im_idx));
        delta_pool_sqz = squeeze(delta_upsampled(:,:,f_idx,im_idx));
        cur_grad = conv2(im, rot90(delta_pool_sqz, 2), 'valid');
        
        Wc_grad(:,:,f_idx) = Wc_grad(:,:,f_idx) + cur_grad;
        bc_grad(f_idx) = bc_grad(f_idx) + sum(delta_pool_sqz(:));
    end
end

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
