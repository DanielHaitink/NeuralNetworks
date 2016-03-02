% mlp.m Implementation of the Multi-Layer Perceptron

clear all
close all

examples = [0 0;1 0;0 1;1 1];
goal = [0.01 0.99 0.99 0.01]';

% Boolean for plotting the animation
plot_animation = true;

% Parameters for the network
learn_rate = 0.2;               % learning rate
max_epoch = 5000;              % maximum number of epochs
min_error = 0.01;

mean_weight = 0;
weight_spread = 2;

n_input = size(examples,2);
n_hidden = 20;
n_output = size(goal,2);

% Noise level at the input
noise_level = 0.01;

% Activation of the bias node
bias_value = -1;


% Initializing the weights
w_hidden = rand(n_input + 1, n_hidden) .* weight_spread - weight_spread/2 + mean_weight;
w_output = rand(n_hidden, n_output) .* weight_spread - weight_spread/2 + mean_weight;

% Start training
stop_criterium = 0;
epoch = 0;

while ~stop_criterium
    epoch = epoch + 1;
    
    % Add noise to the input data.
    noise = randn(size(examples)) .* noise_level;
    input_data = examples + noise;
    
    % Append bias to input data
    input_data(:,n_input+1) = ones(size(examples,1),1) .* bias_value;
    
    epoch_error = 0;
    epoch_delta_hidden = 0;
    epoch_delta_output = 0;
    
    % FROM HEREON YOU NEED TO MODIFY THE CODE!
    for pattern = 1:size(input_data,1)
        
        % Compute the activation in the hidden layer
        hidden_activation = input_data(pattern, :) * w_hidden;
        
        % Compute the output of the hidden layer (don't modify this)
        hidden_output = sigmoid(hidden_activation);
        
        % Compute the activation of the output neurons
        output_activation = hidden_output * w_output;
        
        % Compute the output
        output = output_function(output_activation);
        
        % Compute the error on the output
        output_error = goal(pattern) - output;
        
        % Compute local gradient of output layer
        local_gradient_output = d_output_function(output_activation) .* output_error;
        
        % Compute the error on the hidden layer (backpropagate)
        hidden_error = output_error * w_output;        
        
        % Compute local gradient of hidden layer
        local_gradient_hidden = d_sigmoid(hidden_activation) .* (local_gradient_output * w_output)';
        
        % Compute the delta rule for the output
        delta_output = learn_rate * local_gradient_output' * hidden_output ;
        
        % Compute the delta rule for the hidden units;
        delta_hidden = learn_rate * local_gradient_hidden' * input_data(pattern, :);
        
        % Update the weight matrices
        w_hidden = w_hidden + delta_hidden';
        w_output = w_output + delta_output';
        
        % Store data
        epoch_error = epoch_error + (output_error).^2;        
        epoch_delta_output = epoch_delta_output + sum(sum(abs(delta_output)));
        epoch_delta_hidden = epoch_delta_hidden + sum(sum(abs(delta_hidden)));
    end
    
    % Log data
    h_error(epoch) = epoch_error / size(input_data,1);
    log_delta_output(epoch) = epoch_delta_output;
    log_delta_hidden(epoch) = epoch_delta_hidden;
    
    % Check whether maximum number of epochs is reached
    if epoch > max_epoch
        stop_criterium = 1;
    end
    
    % Implement a stop criterion here
    if min_error >= epoch_error
        stop_criterium = 1;
    end
    
    
    % Plot the animation
    if and((mod(epoch,20)==0),(plot_animation))
        emp_output = zeros(21,21);
        figure(1)
        for x1 = 1:21
            for x2 =  1:21
                hidden_act = sigmoid([(x1/20 - 0.05) (x2/20 -0.05) bias_value] * w_hidden);
                emp_output(x1,x2) = output_function(hidden_act * w_output);
            end
        end
        surf(0:0.05:1,0:0.05:1,emp_output) 
        title(['Network epoch no: ' num2str(epoch)]);
        xlabel('input 1: (0 to 1 step 0.05)')
        ylabel('input 2: (0 to 1 step 0.05)')
        zlabel('Output of network')
        zlim([0 1])        
    end

end

% Plotting the error
figure(2)
plot(1:epoch,h_error)
title('Mean squared error vs epoch');
xlabel('Epoch no.');
ylabel('MSE');

% Add additional plot functions here (optional)
