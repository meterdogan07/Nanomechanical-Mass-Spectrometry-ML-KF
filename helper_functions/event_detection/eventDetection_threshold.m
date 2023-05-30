function [store] = eventDetection_threshold(k, threshold, M, d_yk, ix_max, const, store)    
    if(sum(store.l > threshold) > M/4 && (ix_max < (M/2))) %changed to M/4 and M/2 instead of 15 and 25 respectively
        %time_sample
        k0 = k-(M-ix_max);
        store.detected_event_time_k0(store.counter_for_k0) = k0;
        store.counter_for_k0 = store.counter_for_k0+1;
        store.detected_event_time_k(store.counter_for_k) = k;
        store.counter_for_k = store.counter_for_k+1;
        Fk0 = const.F^(M-ix_max);
        store.y_estimate(:,k) = store.y_estimate(:,k) + (Fk0 - store.Ls2(:,:,ix_max))*d_yk*[1;0];
        store.C(:,:,k) = store.C(:,:,k) + (Fk0 - store.Ls2(:,:,ix_max))*inv(store.CIs2(:,:,ix_max))*(Fk0 - store.Ls2(:,:,ix_max))';
        store.ct = 0;
    end
    
    if store.counter_for_k == 1
        store.detected_event_time_k0 = 0;
        store.detected_event_time_k = 0; 
    end
end