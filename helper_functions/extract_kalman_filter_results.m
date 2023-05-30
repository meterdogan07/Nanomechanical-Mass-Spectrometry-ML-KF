function [extracted_struct] = extract_kalman_filter_results(store, nots, M)
    dyks_plus_y_estimate = store.y_estimate(1,:);
    %Ignore event detection when detection time is zero
    if length(store.detected_event_time_k0)>1
        if store.detected_event_time_k0(1)==0 
            n = length(store.detected_event_time_k0);
            store.detected_event_time_k0 = store.detected_event_time_k0(2:n);
            store.detected_event_time_k = store.detected_event_time_k(2:n);
        end
    elseif length(store.detected_event_time_k0)==1
        if store.detected_event_time_k0(1)==0 
            store.detected_event_time_k0 = [];
            store.detected_event_time_k = [];
        end
    end

    if length(store.detected_event_time_k0)>1 %if there are more than 1 events detected
        false_detection = 1;
        one_detection=0;
        no_detection=0;
        correct=0;
        square_error=0;
        store.detected_event_time_k=0;
        store.detected_event_time_k0=0;
        event_size = 0;
        event_size_after_delay = 0;

    elseif length(store.detected_event_time_k0)==1 % if exactly one event is found
        false_detection = 0;
        one_detection=1;
        no_detection=0;
        for i = 1:length(store.detected_event_time_k)
            dyks_plus_y_estimate(store.detected_event_time_k0(i):store.detected_event_time_k(i)-1) = dyks_plus_y_estimate(store.detected_event_time_k0(i)-1) + store.dyks(store.detected_event_time_k0(i)+1:store.detected_event_time_k(i));
        end
        correct = (nots/2 == store.detected_event_time_k0);
        square_error = (store.detected_event_time_k0 - nots/2)^2;
        event_size = dyks_plus_y_estimate(store.detected_event_time_k0);
        event_size_after_delay = dyks_plus_y_estimate(end); %store.detected_event_time_k0 + 10*M
    else %event is not found
        false_detection = 0;
        one_detection=0;
        no_detection=1;
        correct=0;
        square_error=0;
        store.detected_event_time_k=0;
        store.detected_event_time_k0=0;
        event_size = 0;
        event_size_after_delay = 0;
    end

    extracted_struct.false_detection = false_detection;
    extracted_struct.no_detection = no_detection;
    extracted_struct.event_size = event_size;
    extracted_struct.event_size_after_delay = event_size_after_delay;
    extracted_struct.square_error = square_error;
    extracted_struct.detected_event_time_k = store.detected_event_time_k;
    extracted_struct.detected_event_time_k0 = store.detected_event_time_k0;
    extracted_struct.correct = correct;
    extracted_struct.one_detection = one_detection;
    extracted_struct.time_elapsed = store.time_elapsed;
end