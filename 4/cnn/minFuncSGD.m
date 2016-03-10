function [opttheta] = minFuncSGD(funObj,theta,data,labels,...
                        options,filterDim,numFilters,poolDim,numClasses)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9


%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
        'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
m = length(labels); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));

%%======================================================================
%% SGD loop
it = 0;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end;

        % get next randomly selected minibatch
        mb_data = data(:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        [cost, grad, preds, activations] = funObj(theta,mb_data,mb_labels);
        
        % Instructions: Add in the weighted velocity vector to the
        % gradient evaluated above scaled by the learning rate.
        % Then update the current weights theta according to the
        % sgd update rule
        velocity = (mom .* velocity) + (alpha .* grad);
        theta = theta - velocity;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        
        [Wc] = cnnParamsToStack(theta,size(mb_data,1),filterDim,numFilters,...
                        poolDim,numClasses);
        assert(numFilters==20, 'cannot display weights');
        
       
        tiles = [];
        for i=0:3
            im = squeeze(Wc(:,:,i*5+1));
            im = (im - min(im(:))) ./ (max(im(:)) - min(im(:)));
            row = [im];
            for j=2:5
                im = squeeze(Wc(:,:,i*5+j));
                im = (im - min(im(:))) ./ (max(im(:)) - min(im(:)));
                row = [row, zeros(size(row,1), 1), im];
            end
            tiles = [tiles; row];
            if i < 3
                tiles = [tiles; zeros(1,size(tiles,2))];
            end
        end
        figure(1);
        imshow(tiles, [0, 1], 'initialMagnification', 600);
        title('Filters');
        %zoom on;
        
        
        if mod(it-1,5) == 0
            figure(2);
            subplot(2,1,1);
            im_idx = randi(256);
            original_image = mb_data(:,:,im_idx);
            imshow(original_image, 'initialMagnification', 200);
            title('Original image');
            
            features = squeeze(activations(:,:,:,im_idx));
            subplot(2,1,2);
            tiles = num2cell(features,[1 2]);
            tiles = reshape(tiles,4,5);
            out = cell2mat(tiles);
            imshow(out, [0, 1], 'initialMagnification', 400);
            title(sprintf('Convolved features, prediction: %i', mod(preds(im_idx),10)));
        end
        
    end;

    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;

end;

opttheta = theta;

end
