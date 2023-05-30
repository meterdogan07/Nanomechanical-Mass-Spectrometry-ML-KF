function [dataset] = data_loader(M,event_ratio,params)
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

    size = 100000;
    F = [1,0; h/tau_r, 1-(h/tau_r)];
    nots = M; %number_of_time_samples
    
    dys = zeros(size,1);
    events = zeros(size,1);
    locs = zeros(size,1);
    data = zeros(size,nots);
    
    %DR= 60;
    %SNR = 10^(DR/20);

    fprintf("- bruteforce dataset with event ratio: %d, total simulations: %d \n",event_ratio, size)
    for j = 1:size
        if(rem(j,1000)==0)
            fprintf("data creation with event ratio: %d, simulation number: %d \n",event_ratio, j)
        end
        y_e = zeros(1,nots); %the root cause due to an event of interest
        y_r = zeros(1,nots); %the response of the resonator
        y_ro = zeros(1,nots); %the observation
        y_e(1) = 0;
        y_r(1) = 0;
        y_state = zeros(2,nots);
        y_state(1,:) = y_e;
        y_state(2,:) = y_r;
        
        %IMPULSE (STICKING EVENT)
        if M == 50 %if M=50
            dy = 10^(-3+log10(5)-3*rand); % log dist. of event size between 1e-3<=>1e-6 
        else %if M=10
            dy = 10^(-2-3*rand);%5 * 10^(-3-(2 +log10(5))*rand);
        end
        dys(j) = dy;
        
        u_e = zeros(1,nots);
        event = 0;
        if(rand<event_ratio)
            loc = (M/10)+randi(M-2*(M/10));
            %loc2 = 1050+randi(900);
            locs(j) = loc;
            event = 1;
            u_e(loc) = dy;
            %u_e(loc2) = dy;
        end
    %     if(rand>0)
    %         loc2 = 5+randi(40);
    %         locs(j) = loc;
    %         event = 1;
    %         u_e(loc2) = dy;
    %     end
        events(j) = event; 
        
        % NOISE
        y_th = sqrt(h*S_yth)*randn(1,nots);
        y_d = sqrt(BW_L * S_yd)*randn(1,nots);
        H = [0 1];
        
        %start iteration
        for i=1:nots
            y_state(:,i+1) = F * y_state(:,i) + [u_e(i);0] + [0;y_th(i)/tau_r];
            y_ro(i) = H*y_state(:,i) + y_d(i);
        end
        y_ro(nots) = [0,1]*y_state(:,nots) + y_d(nots);
        
        data(j,:) = y_ro;
        %data(j,:) = y_ro(1001:2000);
    end
    
    dataset = [normalize(data,2) locs dys events];
    %save newdtable_0.1event_50window.mat dataset
end