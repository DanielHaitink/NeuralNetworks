% Returns the calculated t and v in a vector. This can be used to plot the
% points in a figure
function [t,v] = plotNeuron(tau, Rin, theta)
    nstep = 100; % Number of timesteps to integrate over
    Inoise = 0.1;
    I0 = 1+Inoise*randn(1,nstep); % Input current in nA
    dt = 1; % time step in ms
    v = zeros(1,nstep);
    tspike = [];
    t = (1:nstep)*dt;
    for n=2:nstep
        v(n) = v(n-1) + dt*(- v(n-1)/tau + Rin*I0(n)/tau);
        if (v(n) > theta)
            v(n) = 0;
            tspike = [ tspike t(n) ];
        end
    end
end