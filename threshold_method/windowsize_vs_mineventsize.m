h = 1; %time step
w_r = 2*pi; %resonance frequency
tau_r = 1e5; %2*Q/w_r; %inherent resonator time constant
Kd2 = 1e-5;
BW_L = 3e-5; %w_r/Q;
% NOISE
S_yth = 1e-10;
S_yd = S_yth*Kd2;


te_s = 5:500;
y_min = zeros(1,size(te_s,2));
M_s = zeros(1,size(te_s,2));

i=1;
for te = te_s
    Z = S_yth*te + 4*BW_L*S_yd*(tau_r^2);
    var = (Z+sqrt(S_yth*te*Z))/(2*te*te);
    y_min(i) = 3*sqrt(var);
    M_s(i) = 2*te;
    i=i+1;
end

loglog(y_min, M_s, "linewidth", 4)
set(gca,"FontSize",25)
grid on
ylabel("{\it M}",'FontSize', 35, 'FontName', "Times")
xlabel("{\Delta}{\it y_{min}}",'FontSize', 35, 'FontName', "Times")
%xlabel("{\Delta}y_{min}", 'FontSize', 20)
title("Window Size vs. Minimum Detectable Event Size",'FontSize', 35, 'FontName', "Times")

