function [store, window_start] = run_blackbox_simulation(cnet_name, rnet_name, c_net, r_net, y_ro, M, eventPlace)
    fprintf("Blackbox-ML - %s method.\n",cnet_name)
    run('initialize_kalman_variables.m');
    no_detection = 1;
    false_detection = 0;
    event_count = 0;
    total_dy = 0;
    CONF_THRESH = M/2;

    tic
    for k=2:nots-M
        windowx = y_ro(k:k+M-1);
        event = bruteforce_classify(cnet_name,c_net,normalize(windowx,2),M);
        if(event>=0.5)
            reg = bruteforce_regression(rnet_name,r_net,normalize(windowx,2),M);            
            event_count = event_count+1;
            total_dy = total_dy + reg(1);
            if(event_count >= CONF_THRESH)
                event_size = total_dy/(1e5*event_count); %1e5 nereden geliyor?
                detected_time = k+round(reg(2));
                if detected_time <= eventPlace-M
                    false_detection = 1;
                end
                no_detection = 0;
                window_start = k;
                break
            end
        else
            total_dy = 0;
            event_count = 0;
        end
    end

    if no_detection == 1 %if event is not detected
        correct = 0; %event is not found
        square_error = 0; %event is not found
        event_size = 0;
        detected_time = 0;
        window_start = 0;
    end

    time_elapsed = toc;

    store.event_count = event_count;
    store.time_elapsed = time_elapsed;
    store.detected_time = detected_time;
    store.no_detection = no_detection;
    store.event_size = event_size;
    store.false_detection = false_detection;
    store.window_start = window_start;
end