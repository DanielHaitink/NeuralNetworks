% TLU implementation
% Daniel Haitink & Job Talle

% Matrix
inputTable = [0,0; 0,1; 1,0; 1,1];
input1 = [0.1,0.1; 0.1,0.9; 0.9,0.1; 0.9,0.9];
input2 = [0.2,0.2; 0.2,0.8; 0.8,0.2; 0.8,0.8];
andSolution = [0; 0; 0; 1];
orSolution = [0; 1; 1; 1];
nandSolution = [1;1;1;0];

% Parameters
learn_rate = 0.1;    % the learning rate
n_epochs = 100;      % the number of epochs we want to train

% Define the inputs
examples = inputTable;

% Define the corresponding target outputs
goal = andSolution;

% Initialize the weights and the threshold
weights = randi([0 1], 1, 2);
threshold = 0; 

% Preallocate vectors for efficiency. They are used to log your data 
% The 'h' is for history
h_error = zeros(n_epochs,1);
h_weights = zeros(n_epochs,2);
h_threshold = zeros(n_epochs,1);

% Store number of examples and number of inputs per example
n_examples = size(examples,1);     % The number of input patterns
n_inputs = size(examples,2);       % The number of inputs

for epoch = 1:n_epochs
    epoch_error = zeros(n_examples,1);
    
    h_weights(epoch,:) = weights;
    h_threshold(epoch) = threshold;
    
    for pattern = 1:n_examples
        
        % Initialize weighted sum of inputs
        summed_input = examples(pattern,:)*weights';
        
        % Subtract threshold from weighted sum
        summed_input = summed_input-threshold;
        % Compute output
        if summed_input >= 0 
           output = 1; 
        else
           output = 0;
        end
       
        
        % Compute error
        error = output - goal(pattern);
        q = learn_rate * (goal(pattern)-output);
        % Compute delta rule
        delta_weights = q*examples(pattern,:);
        delta_threshold = -q;
        
        % Update weights and threshold
        weights = weights + delta_weights;
        threshold = threshold + delta_threshold;        
    
        % Store squared error
        epoch_error(pattern) = error.^2;
    end
    
    h_error(epoch) = sum(epoch_error);
end

% Plot functions
figure(1);
plot(h_error)
title('\textbf{TLU-error over epochs}', 'interpreter', 'latex', 'fontsize', 12);
xlabel('\# of epochs', 'interpreter', 'latex', 'fontsize', 12)
ylabel('Summed Squared Error', 'interpreter', 'latex', 'fontsize', 12)

figure(2);
plot(1:n_epochs,h_weights(:,1),'r-','DisplayName','weight 1')
hold on
plot(1:n_epochs,h_weights(:,2),'b-','DisplayName','weight 2')
plot(1:n_epochs,h_threshold,'k-','DisplayName','threshold')
xlabel('\# of epochs', 'interpreter', 'latex', 'fontsize', 12)
title('\textbf{Weight vector and threshold vs epochs}', 'interpreter', 'latex', 'fontsize', 12);
h = legend('location','NorthEast');
set(h, 'interpreter', 'latex', 'fontsize', 12);
hold off
