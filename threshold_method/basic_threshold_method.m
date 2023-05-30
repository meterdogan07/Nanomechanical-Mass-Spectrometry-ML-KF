%CONSTANTS
h = 1; %time step
w_r = 2*pi; %resonance frequency
tau_r = 1e5; %2*Q/w_r; %inherent resonator time constant
Kd2 = 1e-5;

%-------------------------------------------------------
% create F2 before for faster simulation
n = 2*200;
F2 = zeros(2,2,n);
F = [1,0; h/tau_r, 1-(h/tau_r)];
for i=1:n
    F2(:,:,i) = F^(i);
end

%-------------------------------------------------------
nots = 2e4; %number_of_time_sample
y_e = zeros(1,nots); %the root cause due to an event of interest
y_r = zeros(1,nots); %the response of the resonator
y_ro = zeros(1,nots); %the observation
y_e(1) = 0;
y_r(1) = 0;
y_state = zeros(2,nots);
y_state(1,:) = y_e;
y_state(2,:) = y_r;
C = zeros(2,2,nots);
C(:,:,1) = zeros(2);

%IMPULSE (STICKING EVENT)
dy = 1e-4; 
u_e = zeros(1,nots);
u_e(nots/2) = dy;
u_e(nots/4) = dy;

% NOISE
S_yth = 1e-10; 
BW_L = 3e-5; 
S_yd = Kd2*S_yth;
y_th = sqrt(h*S_yth)*randn(1,nots);
y_d = sqrt(BW_L * S_yd)*randn(1,nots);
H = [0 1];

%start iteration
for i=1:nots % number of time steps (nots)
    y_state(:,i+1) = F * y_state(:,i) + [u_e(i);0] + [0;y_th(i)/tau_r];
    y_ro(i) = H*y_state(:,i) + y_d(i);
end
y_ro(nots) = [0,1]*y_state(:,nots) + y_d(nots);

%KALMAN FILTERING 
Q = [0 0;0 h*S_yth/(tau_r*tau_r)];
R = 3.0000e-20; 
values = zeros(1,nots);
jump_size = zeros(1,nots);
jump_location = zeros(1,nots);
dyks = zeros(1,nots);
detected_event_time = [0,0];
y_estimate = zeros(2,nots);

M = 50; %window size
threshold = 20; %threshold value
ct = 0;

dd = zeros(2,M,nots);
LL = zeros(2,2,M,nots);
CC = zeros(2,2,M,nots);

d = zeros(2,M);
Ls = zeros(2,2,M);
CIs = zeros(2,2,M);
a = zeros(1,M);
b = zeros(1,M);
l = zeros(1,M);
d2 = zeros(2,M);
Ls2 = zeros(2,2,M);
CIs2 = zeros(2,2,M);

k=1;
while(k<nots)
    k=k+1;
    %predict
    y_estimate(:,k) = F*y_estimate(:,k-1);
    C(:,:,k) = F*C(:,:,k-1)*(F') + Q;
    %observe
    yk = y_ro(k) - H*y_estimate(:,k);
    V = H*C(:,:,k)*(H') + R;
    %Kalman Gain
    K = C(:,:,k)*(H')/V;
    %Estimate
    y_estimate(:,k) = y_estimate(:,k) + K*yk;
    %Estimate Covariance
    C(:,:,k) = (eye(2) - K*H)*C(:,:,k);

    %EVENT DETECTION: UPDATING C MATRIX
%--------------------------------------
    ct = ct+1;
    for m=k-1:-1:max(k-M+1,1)
        ix = M+m-k;
        G = H*(F2(:,:,k-m) - F*Ls(:,:,ix)); %1x2
        CIs2(:,:,ix) = (G')*G/V + CIs(:,:,ix);
        d2(:,ix) = G'*yk/V + d(:,ix);
        Ls2(:,:,ix) = K*G + F*Ls(:,:,ix);
        a(ix) = [1 0]*CIs2(:,:,ix)*[1;0];
        b(ix) = [1 0]*d2(:,ix);
        l(ix) = (b(ix)*b(ix))/a(ix);
    end
    
    [value,ix_max] = max(l);
    d_yk = b(ix_max)/a(ix_max);
    jump_size(k) = d_yk;
    jump_location(k) = k-(M-ix_max);
    dyks(k) = d_yk;
    values(k) = value;
    l = l(1:M-1);

    if(sum(l > threshold) > M/4 && (ix_max < M/2) && ct>M)
        ct = 0; % reset the filter
        detected_event_time(1) = ix_max;
        detected_event_time(2) = k;
        Fk0 = F2(:,:,M-ix_max);
        y_estimate(:,k) = y_estimate(:,k) + (Fk0 - Ls2(:,:,ix_max))*d_yk*[1;0];
        C(:,:,k) = C(:,:,k) + (Fk0 - Ls2(:,:,ix_max))*inv(CIs2(:,:,ix_max))*(Fk0 - Ls2(:,:,ix_max))';
    end

    d2(:,M) = H'*yk/V;
    Ls2(:,:,M) = K*H;
    CIs2(:,:,M) = H'*H/V;

    d(:,1:M-1) = d2(:,2:M);
    Ls(:,:,1:M-1) = Ls2(:,:,2:M);
    CIs(:,:,1:M-1) = CIs2(:,:,2:M);
    dd(:,:,k) = d;
    LL(:,:,:,k) = Ls;
    CC(:,:,:,k) = CIs;

    a = zeros(1,M);
    b = zeros(1,M);
    %--------------------------------------
end

%%
k = detected_event_time(2);
ix = detected_event_time(1);
n=M-ix;
figure
plot(y_state(1,:));
hold on
plot(y_estimate(2,:));
plot(y_estimate(1,:));
legend('y-e','y-r','y-ro','estimate y-r','estimate y-e');
xlabel('time samples');
ylabel('fractional frequency shift');
