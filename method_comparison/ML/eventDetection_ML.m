confidence = 0;

%Combines dyks with y_estimate

fs = 264000;
t = (1:(nots+1))/fs;

%CONSTANTS
h = 1; %time step
w_r = 2*pi; %resonance frequency
tau_r = 1e5; %2*Q/w_r; %inherent resonator time constant
Kd2 = 1e-5;

%DR= 60;
%SNR = 10^(DR/20);
F = [1,0; h/tau_r, 1-(h/tau_r)];

nots = 2e3; %number_of_time_samples
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
u_e(nots/4-1) = dy;
u_e(nots/2-1) = dy;

% NOISE
S_yth = 1e-10; %1/(4*SNR*SNR*BW_L);%kb*T/(m*Q*(w_r^3)*Arss*Arss);
BW_L = 3e-5; %w_r/Q;
S_yd = Kd2*S_yth;
y_th = sqrt(h*S_yth)*randn(1,nots);
y_d = sqrt(BW_L * S_yd)*randn(1,nots);     
H = [0 1];

%start iteration
for i=1:nots
    y_state(:,i+1) = F * y_state(:,i) + [u_e(i);0] + [0;y_th(i)/tau_r];
    y_ro(i) = H*y_state(:,i) + y_d(i);
end
y_ro(nots) = [0,1]*y_state(:,nots) + y_d(nots);

%KALMAN FILTERING 
Q = [0 0;0 h*S_yth/(tau_r*tau_r)];
R = [BW_L*S_yd];
values = zeros(1,nots);
dyks = zeros(1,nots);
detected_event_time_k0 = [];
counter_for_k0=1;
detected_event_time_k = [];
counter_for_k=1;
y_estimate = zeros(2,nots);

M = 50; %50
threshold = 3000;
ct = 0;

d = zeros(2,M);
Ls = zeros(2,2,M);
CIs = zeros(2,2,M);
a = zeros(1,M);
b = zeros(1,M);
l = zeros(1,M);
d2 = zeros(2,M);
Ls2 = zeros(2,2,M);
CIs2 = zeros(2,2,M);

ls = zeros(nots, M-1);

figure()
plot(t, y_state(2,:));
grid on
xlabel("Time (s)");
ylabel("Amplitude")
title("State response after an event of relative size dy = 1e-3")

%% -- kalman filter
for k=2:nots
%     if k == nots/4 %EVENT COVARIANCE
%         C(:,:,k-1) = [1e-10/8 0;0 0];
%     end
    %predict
    y_estimate(:,k) = F*y_estimate(:,k-1);
    C(:,:,k) = F*C(:,:,k-1)*(F') + Q;
    %observe
    yk = y_ro(k) - H*y_estimate(:,k);
%    %Kalman Gain
    K = C(:,:,k)*(H')/V;
    %Estimate
    y_estimate(:,k) = y_estimate(:,k) + K*yk;
    %Estimate Covariance
    C(:,:,k) = (eye(2) - K*H)*C(:,:,k);

    %EVENT DETECTION: UPDATING C MATRIX
%--------------------------------------
    % ct = ct+1;
    for m=k-1:-1:k-ct
        ix = M+m-k;

        G = H*(F^(k-m) - F*Ls(:,:,ix)); %1x2
        CIs2(:,:,ix) = (G')*G/V + CIs(:,:,ix);
        d2(:,ix) = G'*yk/V + d(:,ix);
        Ls2(:,:,ix) = K*G + F*Ls(:,:,ix);
        a(ix) = [1 0]*CIs2(:,:,ix)*[1;0];
        b(ix) = [1 0]*d2(:,ix);
        l(ix) = (b(ix)*b(ix))/a(ix);
    end
    if ct >= M-1
        [value,ix_max] = max(l);
        d_yk = b(ix_max)/a(ix_max);
        dyks(k) = d_yk;
        k0 = k-(M-ix_max);
        k0s(k) = k0;
        values(k) = value;
    end
    
    l = l(1:M-1);
    % ls(k,:) = l;
    
    % ENSEMBLE
        %event = predict(ens, l);
    
    %TENSORFLOW
         
          mu = 0.1;
          event = predict(net, l);
          if(event<=mu)
              event = 0;
          else
              event = 1;
          end
              
        
    confidence = (confidence+event)*event;
    
    if(confidence >=10 && ct>=M-1)
        %k
        %k0 = k-(M-ix_max)
        
        fprintf('The window where we detect an event in it is %d.\n', k);
        fprintf('Event is predicted to be at %d.\n', k0);
        detected_event_time_k0(counter_for_k0) = k0;
        counter_for_k0 = counter_for_k0+1;
        detected_event_time_k(counter_for_k) = k;
        counter_for_k = counter_for_k+1;
        Fk0 = F^(M-ix_max);
        y_estimate(:,k) = y_estimate(:,k) + (Fk0 - Ls2(:,:,ix_max))*d_yk*[1;0];
        C(:,:,k) = C(:,:,k) + (Fk0 - Ls2(:,:,ix_max))*inv(CIs2(:,:,ix_max))*(Fk0 - Ls2(:,:,ix_max))';

        ct = 0;
        confidence = 0;
    end

    d2(:,M) = H'*yk/V;
    Ls2(:,:,M) = K*H;
    CIs2(:,:,M) = H'*H/V;

    d(:,1:M-1) = d2(:,2:M);
    Ls(:,:,1:M-1) = Ls2(:,:,2:M);
    CIs(:,:,1:M-1) = CIs2(:,:,2:M);
    
    if ct < M-1
        ct = ct+1;
    end
    %--------------------------------------
end

%GET the combination of dyks and y_estimate
dyks_plus_y_estimate = y_estimate(1,:);
for i2 = 1:length(detected_event_time_k)
    dyks_plus_y_estimate(detected_event_time_k0(i2):detected_event_time_k(i2)-1) = dyks_plus_y_estimate(detected_event_time_k0(i2)-1) + dyks(detected_event_time_k0(i2)+1:detected_event_time_k(i2));
end


% k = detected_event_time(2);
% ix = detected_event_time(1);
% n=M-ix;
figure
plot(t, y_state(1,:));
hold on
grid on
%plot(y_state(2,:));
% % % plot(y_ro);
% % % plot(y_estimate(2,:));
plot(t(1:end-1), dyks_plus_y_estimate);
% plot(t(1:end), y_state(2, :));
%plot(y_estimate(1,:));
legend('Step disturbance','Estimate of the event size', 'Original State Response');
title("Prediction for Event Amplitude")
xlabel('Time(s)');
ylabel('Fractional Frequency Shift');
