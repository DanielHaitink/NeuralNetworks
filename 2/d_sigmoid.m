%this functions calculates the differential of the sigmoid
function [output] = d_sigmoid(x)
    temp = sigmoid(x);
    output = temp .* (1 - temp);
end