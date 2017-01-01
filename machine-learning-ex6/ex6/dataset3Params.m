function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

SAMPLES = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
SAMPLES_LENGTH = length(SAMPLES);

% Initialize the reference error
err = -999;

% Initialize the current values
C_curr = 0;
sigma_curr = 0;
err_curr = 0;

for i=1:SAMPLES_LENGTH
    % Get the C value
    C_curr = SAMPLES(i);

    % Loop for all the possible sigmas
    for j=1:SAMPLES_LENGTH
        sigma_curr = SAMPLES(j);

        % Create "short hand" for the cost function to be minimized
        kernel = @(x1, x2) gaussianKernel(x1, x2, sigma_curr);

        % Step:1 Build the model
        model = svmTrain(X, y, C_curr, kernel);

        % Step:2 Predict the labels on the cross-validation set
        pred = svmPredict(model, Xval);

        % Step:3 Calculate the error due to misclassification
        err_curr = mean(double(pred ~= yval));

        % If this is the first iteration then set the current error as the reference
        if (err == -999)
            err = err_curr;
        endif

        % Step:4 Check if this error is the least so far
        if (err_curr < err)
            % So set this as the best
            C = C_curr;
            sigma = sigma_curr;

            % Update the baseline
            err = err_curr;
        endif
    end
end



% =========================================================================

end
