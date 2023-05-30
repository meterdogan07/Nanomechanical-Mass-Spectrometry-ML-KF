function [confidence, store] = kalmanFiltering_ver(k, confidence, conf_th, method, y_ro, net, M, const, store)
    %predict
    store.y_estimate(:,k) = const.F*store.y_estimate(:,k-1);
    store.C(:,:,k) = const.F*store.C(:,:,k-1)*(const.F') + const.Q;
    %observe
    yk = y_ro(k) - const.H*store.y_estimate(:,k);
    const.V = const.H*store.C(:,:,k)*(const.H') + const.R;
    %Kalman Gain
    const.K = store.C(:,:,k)*(const.H')/const.V;
    %Estimate
    store.y_estimate(:,k) = store.y_estimate(:,k) + const.K*yk;
    %Estimate Covariance
    store.C(:,:,k) = (eye(2) - const.K*const.H)*store.C(:,:,k);

    % likelihood calculation
    for m=k-1:-1:k-store.ct
        ix = M+m-k;

        const.G = const.H*(const.F^(k-m) - const.F*store.Ls(:,:,ix)); %1x2
        store.CIs2(:,:,ix) = (const.G')*const.G/const.V + store.CIs(:,:,ix);
        store.d2(:,ix) = const.G'*yk/const.V + store.d(:,ix);
        store.Ls2(:,:,ix) = const.K*const.G + const.F*store.Ls(:,:,ix);
        store.a(ix) = store.CIs2(1,1,ix); %[1 0]*CIs2(:,:,ix)*[1;0];
        store.b(ix) = store.d2(1,ix); %[1 0]*d2(:,ix);
        store.l(ix) = (store.b(ix)*store.b(ix))/store.a(ix);
    end
    if store.ct >= M-1
        [value, ix_max] = max(store.l);
        d_yk = store.b(ix_max)/store.a(ix_max);
        store.dyks(k) = d_yk;
        k0 = k-(M-ix_max);
        store.k0s(k) = k0;
        store.values(k) = value;
    
        % event detection
        if(method == "threshold")
            [store] = eventDetection_threshold(k, net, M, d_yk, ix_max, const, store);
        elseif(method == "tensorflow")
            [confidence, store] = eventDetection_tf(k, confidence, conf_th, net, M, d_yk, ix_max, const, store);
        elseif(method == "ensemble")
            [confidence, store] = eventDetection_ens(k, confidence, conf_th, net, M, d_yk, ix_max, const, store);
        elseif(method == "MATLABnn")
            [confidence, store] = eventDetection_MATLAB_nn(k, confidence, conf_th, net, M, d_yk, ix_max, const, store);
        elseif(method == "XGBoost")
            [confidence, store] = eventDetection_xgb(k, confidence, conf_th, net, M, d_yk, ix_max, const, store);
        end
    end

    store.d2(:,M) = const.H'*yk/const.V;
    store.Ls2(:,:,M) = const.K*const.H;
    store.CIs2(:,:,M) = const.H'*const.H/const.V;

    store.d(:,1:M-1) = store.d2(:,2:M);
    store.Ls(:,:,1:M-1) = store.Ls2(:,:,2:M);
    store.CIs(:,:,1:M-1) = store.CIs2(:,:,2:M);
    
    if store.ct < M-1
        store.ct = store.ct+1;
    end
    
end