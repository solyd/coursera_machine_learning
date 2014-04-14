function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters.

  % Initialize some useful values
  m = length(y); % number of training examples

  % You need to return the following variables correctly
  %J = 0;
  %grad = zeros(size(theta));

  h = h_theta(theta, X);

  non_regularized_J = (1/m) * sum((log(h)' * (-y)) -
                                  (log(1 .- h)' * (1 - y)));
  J = non_regularized_J + ((lambda / (2*m)) * sum(theta(2:size(theta)(1), :) .^ 2));

  non_regularized_grad = (1/m) * ((h_theta(theta, X) - y)' * X)';
  grad = non_regularized_grad + ((lambda / m) * theta);
  grad(1) = non_regularized_grad(1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
