function [data, label] = generate_train_data_kalman(M, params)
    nots = M*20; %number_of_time_samples
    F_data_generator;
    
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
    C = zeros(2,2,nots);
    C(:,:,1) = zeros(2);
    
    realization_number = 1e4/M;
    ls = zeros(realization_number, nots-M, M-1);
    label = zeros(realization_number, nots-M);
    
    fprintf("- dataset being created, total simulations: %d \n", realization_number)
    for rn=1:realization_number 
        fprintf("data creation, simulation number: %d \n", rn)
        %IMPULSE (STICKING EVENT)
        if M == 50 %if M=50
            % dy = 10^(-2+log10(5)-4*rand); % xnewdatav4
            % dy = 7.5e-6; % xnewdatav3
            % dy = 10^(-3-3*rand); % xnewdatav2
            % dy = 10^(-5+log10(5)-1*rand); % xnewdata 
            dy = 10^(-3+log10(5)-3*rand); %7e-6 + (2e-6)*rand; %log dist. of event size between 1e-3<=>1e-6 
        else %if M=10
            dy = 10^(-2-3*rand);%5 * 10^(-3-(2 +log10(5))*rand);
        end
        u_e = zeros(1,nots);
        loc = M + randi(nots-2*M);
        u_e(loc-1) = dy;

        % IGNORE FIRST 10 SAMPLES (start labelling only at the 6th sample)
        %label(rn,loc-M+1:loc) = 1;
        label(rn,loc-M+(M/10):loc-(M/10)) = 1;
    
        % NOISE
        y_th = sqrt(h*S_yth)*randn(1,nots);
        y_d = sqrt(BW_L * S_yd)*randn(1,nots);     
        H = [0 1];
    
        %DATA GENERATION
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
        detected_event_time = [0,0];
        y_estimate = zeros(2,nots);
        
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
        
        for k=2:nots
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
        dyks(k) = d_yk;
        values(k) = value;
        if k>M
            ls(rn,k-M,:) = l(1:M-1);
        end
        d2(:,M) = H'*yk/V;
        Ls2(:,:,M) = K*H;
        CIs2(:,:,M) = H'*H/V;
    
        d(:,1:M-1) = d2(:,2:M);
        Ls(:,:,1:M-1) = Ls2(:,:,2:M);
        CIs(:,:,1:M-1) = CIs2(:,:,2:M);
    
        %--------------------------------------
        end
    end
    
    ls = stacker(ls);
    label = reshape(label', (nots-M)*realization_number, 1);
    [data, label] = fixer(ls, label, nots-M);
    
    anso_orig = ls(realization_number,:,:);
    anso_orig = squeeze(anso_orig);
    label_stacked = reshape(label',[],1);
end