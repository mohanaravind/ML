function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Step:1 Compute the prediction(hypohesis)
h = X * theta;

% Step:2 Compute the unregularized cost
unregularized_cost = (1/(2 * m)) * sumsq(h -y);

% Step:3 Calculate the regularization (to minimize overfitting)
% we need not regularize the first term (constant)
no_of_rows = size(theta)(1);
regularization = (lambda/(2 * m)) * sumsq(theta(2:no_of_rows));

% Step:4 Calculate the regularized cost
J = unregularized_cost + regularization;


%---------------------
%----Gradient calc-----
%---------------------
% for the unregularized part of theta (compute how much
% it is responsible for the deviation of hypohesis from actual output)
grad(1) = (1/m) * sum((h - y) * X(1));

% Calculate gradient for other theta's (regularized part)
no_of_cols = size(X)(2);
grad(2:no_of_rows) = (1/m) * (X(:,2:no_of_cols)' * (h-y)) + ((lambda/m) * theta(2:no_of_rows));

% =========================================================================

grad = grad(:);

end
