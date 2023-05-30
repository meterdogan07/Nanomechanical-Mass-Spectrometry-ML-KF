function [y_ro,y_state] = generata_sensor_data(nots,eventPlace,dy,params)
    %CONSTANTS
%     h = 1; %time step
%     w_r = 2*pi; %resonance frequency
%     tau_r = 1e5; %2*Q/w_r; %inherent resonator time constant
%     Kd2 = 1e-5;
%     S_yth = 1e-10; %1/(4*SNR*SNR*BW_L);%kb*T/(m*Q*(w_r^3)*Arss*Arss);
%     BW_L = 3e-5; %w_r/Q
    
    h = params.h;
    w_r = params.w_r;
    tau_r = params.tau_r;
    Kd2 = params.Kd2;
    S_yth = params.S_yth;
    BW_L = params.BW_L;
    S_yd = Kd2*S_yth;

    F = [1,0; h/tau_r, 1-(h/tau_r)];
	y_e = zeros(1,nots); %the root cause due to an event of interest
	y_r = zeros(1,nots); %the response of the resonator
	y_ro = zeros(1,nots); %the observation
	y_e(1) = 0;
	y_r(1) = 0;
	y_state = zeros(2,nots);
	y_state(1,:) = y_e;
	y_state(2,:) = y_r;

	%IMPULSE (STICKING EVENT)
	u_e = zeros(1,nots);
	%u_e(nots/4-1) = dy;
	u_e(eventPlace-1) = dy;

	% NOISE
	S_yd = Kd2*S_yth;
	y_th = sqrt(h*S_yth)*randn(1,nots);
	y_d = sqrt(BW_L * S_yd)*randn(1,nots);     
	H = [0 1];

	%Data generation
	for i=1:nots
		y_state(:,i+1) = F * y_state(:,i) + [u_e(i);0] + [0;y_th(i)/tau_r];
		y_ro(i) = H*y_state(:,i) + y_d(i);
	end
	y_ro(nots) = [0,1]*y_state(:,nots) + y_d(nots);
end
