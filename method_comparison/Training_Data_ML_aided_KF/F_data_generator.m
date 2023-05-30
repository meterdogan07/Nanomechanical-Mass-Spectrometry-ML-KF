%F_data_generator

%CONSTANTS
h = 1; %time step
w_r = 2*pi; %resonance frequency
tau_r = 1e5; %2*Q/w_r; %inherent resonator time constant
Kd2 = 1e-5;

%DR= 60;
%SNR = 10^(DR/20);
F = [1,0; h/tau_r, 1-(h/tau_r)];

F2 = zeros(2,2,nots);
for i_counter=1:nots
    F2(:,:, i_counter) = F^(i_counter);
end