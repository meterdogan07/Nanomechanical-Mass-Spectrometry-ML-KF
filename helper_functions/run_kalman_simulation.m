function [store] = run_kalman_simulation(method, y_ro, net, M, conf_th) 
    fprintf("Kalman Filter based - %s method.\n",method)
    run('initialize_kalman_variables.m');
    confidence = 0;
    tic
    for k=2:nots
        [confidence, store] = kalmanFiltering_ver(k, confidence, conf_th, method, y_ro, net, M, const, store);
    end
    time_elapsed = toc;
    store.time_elapsed = time_elapsed;
end