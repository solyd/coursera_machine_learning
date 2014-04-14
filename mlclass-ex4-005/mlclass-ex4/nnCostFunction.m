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
                   hidden_layer_size, (input_layer_size + 1)); % 25x401

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1)); % 10x26

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

  Y_mat = (y == 1);
  for k = 2:num_labels
    Y_mat = [Y_mat (y == k)];
  end
  % Y_mat is 10x5000

  input_layer_input = X; % 5000x400
  input_layer_input = [ ones(size(input_layer_input, 1), 1) input_layer_input ]; % 5000x401
  input_layer_output = input_layer_input; % 5000x401

  hidden_layer_input = Theta1 * input_layer_output'; % 25x401 * 401x5000 => 25x5000
  hidden_layer_output = sigmoid(hidden_layer_input); % 25x5000
  hidden_layer_output = [ ones(1, size(hidden_layer_output, 2)) ; hidden_layer_output ]; % 26x5000

  output_layer_input = Theta2 * hidden_layer_output; % 10x26 * 26x5000 => 10x5000
  output_layer_output = sigmoid(output_layer_input)'; % 5000x10
  output_layer_output_T = output_layer_output'; % 10x5000

  non_regularized_J = (1/m) * ...
                      sum(diag((-Y_mat) * log(output_layer_output_T) - ...
                               (1 - Y_mat) * log(1 - output_layer_output_T)));

  Theta1_without_bias = Theta1(:, 2:end); % 25x400
  Theta2_without_bias = Theta2(:, 2:end); % 10x25

  J = non_regularized_J + ...
      (lambda/(2*m)) * (sum((Theta1_without_bias .^ 2)(:)) + sum((Theta2_without_bias .^ 2)(:)));

  delta_output_layer = output_layer_output - Y_mat; % 5000x10
  delta_hidden_layer = (Theta2' * delta_output_layer')(2:end, :) ... % 26x10 * 10x5000 => 26x5000 => 25x5000
                       .* sigmoidGradient(hidden_layer_input); % 25x5000

  for i = 1:size(input_layer_output, 1)
    Theta1_grad = Theta1_grad + ...
                  delta_hidden_layer(:, i) * ... % 25x1
                  input_layer_output(i, :); % 1x401 => 25x401
  end

  for i = 1:size(hidden_layer_output, 2)
    Theta2_grad = Theta2_grad + ...
                  delta_output_layer(i, :)' * ... % 10x1
                  hidden_layer_output(:, i)'; % 1x26 => 10x26
  end

  Theta1_grad = (1/m) * Theta1_grad;
  Theta2_grad = (1/m) * Theta2_grad;

  % Regularization of Grad
  Theta1_tmp = Theta1;
  Theta1_tmp(:, 1) = 0;
  Theta2_tmp = Theta2;
  Theta2_tmp(:, 1) = 0;

  Theta1_grad = Theta1_grad + (lambda/m) .* Theta1_tmp;
  Theta2_grad = Theta2_grad + (lambda/m) .* Theta2_tmp;



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

  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
