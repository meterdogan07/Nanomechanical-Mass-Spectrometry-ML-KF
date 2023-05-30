%CONSTANTS
const.Fs = 264e3;
nots = 2e3; %number_of_time_samples
const.t = (0:(nots))/const.Fs;

const.h = 1; %time step
const.w_r = 2*pi; %resonance frequency
const.tau_r = 1e5; %2*Q/w_r; %inherent resonator time constant
const.Kd2 = 1e-5;
const.BW_L = 3e-5; %w_r/Q;
% NOISE
const.S_yth = 1e-10; %1/(4*SNR*SNR*BW_L);%kb*T/(m*Q*(w_r^3)*Arss*Arss);

params.h = const.h;
params.w_r = const.w_r;
params.tau_r = const.tau_r;
params.Kd2 = const.Kd2;
params.S_yth = const.S_yth;
params.BW_L = const.BW_L;

%DR= 60;
%SNR = 10^(DR/20);
const.F = [1,0; const.h/const.tau_r, 1-(const.h/const.tau_r)];
store.C = zeros(2,2,nots);
store.C(:,:,1) = zeros(2);

% %IMPULSE (STICKING EVENT)
% dy = 2e-4; %Event size (fractional freq shift caused by the event)
const.eventPlace = nots/2;

const.S_yd = const.Kd2*const.S_yth;
const.y_th = sqrt(const.h*const.S_yth)*randn(1,nots);
const.y_d = sqrt(const.BW_L * const.S_yd)*randn(1,nots);     
const.H = [0 1];

%KALMAN FILTERING 
const.Q = [0 0;0 const.h*const.S_yth/(const.tau_r*const.tau_r)];
const.R = const.BW_L*const.S_yd;
store.values = zeros(1,nots);

store.dyks = zeros(1,nots);
store.counter_for_k0=1;
store.counter_for_k=1;
store.detected_event_time_k0 = [];
store.detected_event_time_k = [];
store.y_estimate = zeros(2,nots);

store.ct = 0;
store.d = zeros(2,M);
store.Ls = zeros(2,2,M);
store.CIs = zeros(2,2,M);
store.a = zeros(1,M);
store.b = zeros(1,M);
store.l = zeros(1,M);
store.d2 = zeros(2,M);
store.Ls2 = zeros(2,2,M);
store.CIs2 = zeros(2,2,M);
store.time_sample = 0;
store.k0s = [];