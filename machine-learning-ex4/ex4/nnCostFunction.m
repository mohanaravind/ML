function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%A ccumulate error of each output
for i = 1:m
    % i = 1;

    % Get the current data set
    x = X(i, :);

    % Step:1 Compute output at first layer
    % Add the bias
    x = [1 x];
    z = x * Theta1';
    h = sigmoid(z);

    % Step:2 Compute output at second(hidden) layer
    % Add the bias
    h = [1 h];
    z = h * Theta2';
    h = sigmoid(z);

    % Step:3 Actual output for current data set
    output = zeros(10, 1);
    % Set the output label as active
    output(y(i)) = 1;

    %Formula: −ylogh − (1 − y)log(1 − h)
    % Calculate the error for each category(label)
    for k = 1:num_labels
        % Get the data for the current label
        out = output(k);
        pred = h(k);
        J = J + (-out * log(pred) - (1 - out) * log(1-pred));
    end
end

% Cost computation
J = J / m;

% Do regularization (reducing the effect of overfit)
regul = sum(sumsq(Theta1(:, 2:input_layer_size + 1))) + ...
        sum(sumsq(Theta2(:, 2:hidden_layer_size + 1)));
regul = (lambda/(2 * m)) * regul;

% Final cost
J = J + regul;


% backpropagation
% Find out how much of each individual theta is responsible
% for the deviation of hypothesis from actual output.
% i.e How much it is contributing to the value of cost of error
% over all the training data
% Initialize the delta
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for t = 1:m
    % --------------------
    %  Step1: Feedforward
    % --------------------
    % Layer-1 output
    a1 = X(t, :);

    % Layer-2 output
    % Add the bias
    a1 = [1 a1];
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);

    % Layer-3 output
    % Add the bias
    a2 = [1 a2];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    % --------------------------
    %  Step2: calc Output delta
    % ---------------------------
    % Actual data
    output = zeros(num_labels, 1);
    output(y(t)) = 1;
    % The deviation
    d3 = a3' - output;

    % -------------------------------
    %  Step3: Feedbackward the error
    % -------------------------------
    d2 = (d3' * Theta2) .* sigmoidGradient([1 z2]);

    % -------------------------------
    %  Step4: Feedbackward the error
    % -------------------------------
    % Removing the bias
    delta1 = delta1 + d2(2:end)' * a1;
    delta2 = delta2 + d3 * a2;
end

% -------------------------------
%  Step5: Unregularized Gradient
% -------------------------------
Theta1_grad = delta1/m;
Theta2_grad = delta2/m;

% -------------------------------
% Theta1_grad regularization
%--------------------------------
% Account for regularization in gradient calc
% Avoid regularizing the bias
unbiased_length = input_layer_size + 1;
Theta1_grad_to_regul = Theta1_grad(:, 2:unbiased_length);

% Initialize regularization
regul = zeros(size(Theta1_grad_to_regul));
regul = Theta1(:, 2:unbiased_length) * (lambda/m);

Theta1_grad(:, 2:unbiased_length) = Theta1_grad_to_regul + ...
                                    regul;

% -------------------------------
% Theta2_grad regularization
%--------------------------------
% Account for regularization in gradient calc
% Avoid regularizing the bias
unbiased_length = hidden_layer_size + 1;
Theta2_grad_to_regul = Theta2_grad(:, 2:unbiased_length);

% Initialize regularization
regul = zeros(size(Theta2_grad_to_regul));
regul = Theta2(:, 2:unbiased_length) * (lambda/m);

Theta2_grad(:, 2:unbiased_length) = Theta2_grad_to_regul + ...
                                    regul;

% Theta2_grad(:, t)
% size(Theta1_grad) 25x401
% Theta2_grad; 10x26


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
