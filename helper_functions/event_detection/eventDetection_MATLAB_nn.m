function [confidence, store] = eventDetection_MATLAB_nn(k, confidence, CONF_THRESH, net, M, d_yk, ix_max, const, store)
    %CONF_THRESH = 2*M/3;
    %net = importKerasNetwork('./ML/model/kalman_tf.h5', 'OutputLayerType','classification');
    store.l = store.l(1:M-1);
    event = predict(net, reshape(store.l', [1,1,size(store.l,2), size(store.l,1)]));
    event = event(:,2);
    %fprintf("event: %f, k: %f \n", event, k)
    if(event <= 0.1)
        event = 0;
    else
        event = 1;
    end
    
    confidence = (confidence+event)*event;

    if((confidence > CONF_THRESH*M) && (ix_max < (M/2)))
        k0 = k-(M-ix_max);
        store.detected_event_time_k0 = [store.detected_event_time_k0 k0];
        store.counter_for_k0 = store.counter_for_k0+1;
        store.detected_event_time_k = [store.detected_event_time_k k];
        store.counter_for_k = store.counter_for_k+1;
        Fk0 = const.F^(M-ix_max);
        store.y_estimate(:,k) = store.y_estimate(:,k) + (Fk0 - store.Ls2(:,:,ix_max))*d_yk*[1;0];
        store.C(:,:,k) = store.C(:,:,k) + (Fk0 - store.Ls2(:,:,ix_max))*inv(store.CIs2(:,:,ix_max))*(Fk0 - store.Ls2(:,:,ix_max))';
        store.ct = 0;
        confidence = 0;
    end
    if store.counter_for_k == 1
        store.detected_event_time_k0 = 0;
        store.detected_event_time_k = 0; 
    end
end