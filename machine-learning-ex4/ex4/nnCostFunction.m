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

Xnew = [ones(m,1) X]; % 5000 * 401
z2 = Xnew * Theta1';  % 5000 * 25
a2 = sigmoid(z2);     % 5000 * 25 after sigmoid function
a2new = [ones(m,1) a2];  % 5000 * 26 
z3 = a2new * Theta2';  % 5000 * 10
a3 = sigmoid(z3);
ynew = repmat(y, 1, num_labels) == repmat([1:num_labels], m, 1);

% J without regularization
J = (1/m) * sum( sum(-ynew .* log(a3) - (1 - ynew) .* log(1 - a3)) );

% Because we do not do regulatization for theta0, so we delete the frist
% row.
theta1New = Theta1(:, 2: end);
theta2New = Theta2(:, 2: end);

% calculate sum of theta1 then add sum of theta2
reg = lambda/(2*m) * (sum(sum(theta1New.^2)) + sum(sum(theta2New.^2)));
J = J + reg;



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad anqd Theta2_grad. You should return the partial derivatives of
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

for i = 1:m
    a1i = Xnew(i, :); % 1 by 401
    a2i = a2new(i, :);   % 1 by 26
    a3i = a3(i, :);   % 1 by 10
    yi = ynew(i, :);  % 1 by 10
    del3i = a3i - yi; % 1 by 10
    del2i = Theta2' * del3i' .* sigmoidGradient([1; Theta1 * a1i']); % .* (a2i .* (1 - a2i)'); % 26 by 10 * 10 by 1 * 1 by 26 * 1 by 26
    del2i = del2i(2:end); % 25 by 1 
    Theta2_grad = Theta2_grad + del3i' * a2i; % 10 by 1 * 1 by 26 = 10 by 26
    Theta1_grad = Theta1_grad + del2i * a1i; % 25 by 1 * 1 by 401 = 25 by 401
end

Theta2_grad = 1/m * Theta2_grad; % 10 by 26
Theta1_grad = 1/m * Theta1_grad; % 25 by 401



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
