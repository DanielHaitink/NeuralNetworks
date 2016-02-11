clear v;

% calculate the three different vectors which will be plot. The variables
% have to be changed by hand
[t0,v0] = plotNeuron(10,5,4);
[t1,v1] = plotNeuron(10,5,3);
[t2,v2] = plotNeuron(10,5,5);
figure(1)
%plot the first (medium) graph
subplot(3,1,1);
plot(t0,v0, 'b');
title('Theta = 4');
xlabel('Time (ms)');
ylabel('Ion Flow');

%plot the second (low) graph
subplot(3,1,2);
plot(t1,v1, 'm');
title('Theta = 3');
xlabel('Time (ms)');
ylabel('Ion Flow');

%plot the third (high) graph
subplot(3,1,3);
plot(t2,v2, 'c');
title('Theta = 5');
xlabel('Time (ms)');
ylabel('Ion  Flow');
hold off
