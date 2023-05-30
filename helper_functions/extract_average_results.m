function [run_time, False_detection, Miss, Detection_delay, k0, rmse_loc, rmse_size, rmse_size_delay] = extract_average_results(kalman_res_ext, number_of_experiments, event_loc, dy)
% initialize table columns
run_time = [];
False_detection = []; % 8 for each method (5+3)
Miss = [];
Detection_delay = [];
mean_k0 = [];
rmse_loc = [];
rmse_size = [];
rmse_size_delay = []; % NA for blackbox
k0 = [];
k = [];

for i=1:number_of_experiments % each experiment is its own column
    
    run_time = [run_time, [kalman_res_ext(i).time_elapsed]];
    False_detection = [False_detection, [kalman_res_ext(i).false_detection]];
    Miss = [Miss, [kalman_res_ext(i).no_detection]];

    % find which samples have miss or false_detection to find error
    erroneous_detection = (Miss(end) + False_detection(end)) > 0;
    k = [k, [kalman_res_ext(i).detected_event_time_k]];
    k0 = [k0, [kalman_res_ext(i).detected_event_time_k0]];

    Detection_delay = [Detection_delay, abs(k(end)-k0(end))];
    rmse_loc = [rmse_loc, ((k0(end)-event_loc(end)).^2)]; % take sqrt later
    dy_hat = kalman_res_ext(i).event_size;
    rmse_size = [rmse_size, ((dy_hat-dy).^2)]; % take sqrt later
    dy_hat_delay = kalman_res_ext(i).event_size_after_delay;
    rmse_size_delay = [rmse_size_delay, ((dy_hat_delay-dy).^2)]; % take sqrt later
    
    if(erroneous_detection)
        Detection_delay(end) = -1;
        k0(end) = -1;
        rmse_loc(end) = -1;
        rmse_size(end) = -1;
        rmse_size_delay(end) = -1;
    end
end

% post processing
run_time = mean(run_time, 2);
False_detection = sum(False_detection, 2);
False_detection = (100/number_of_experiments) * False_detection; % convert to percent
Miss = sum(Miss, 2);
Miss = (100/number_of_experiments) * Miss; % convert to percent

% selective mean to consider non -1 entries only
k0 = selective_mean(k0);
Detection_delay = selective_mean(Detection_delay);
rmse_loc = sqrt(selective_mean(rmse_loc));
rmse_size = sqrt(selective_mean(rmse_size))/dy;
rmse_size_delay = sqrt(selective_mean(rmse_size_delay))/dy;
