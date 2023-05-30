function [x_train, x_test, y_train, y_test] = test_train_split(X, Y, test_size)

    [train_idx, ~, test_idx] = dividerand(size(Y), test_size, 0, 1-test_size);
    % slice training data with train indexes 
    %(take training indexes in all 10 features)
    x_train = X(train_idx, :);
    y_train = Y(train_idx);
    % select test data
    x_test = X(test_idx, :);
    y_test = Y(test_idx);
end