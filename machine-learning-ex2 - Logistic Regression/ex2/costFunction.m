function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%%
  h = sigmoid(X*theta);
  J = 1/m *((-y)'*log(h) - (ones(length(y),1)-y)'*log(1-h)); % Note log base e
  grad = 1/m * X' * (h-y);

%% Implementation 2 

% for i=1:m
%   h = sigmoid(theta'*X(i,:)');
%   J = 1/m *(-y(i)*log10(h) - (1-y(i))*log10(1-h))
%   grad = grad + (y(i) - h);
% end;

% %% Implementation 1
% grad = zeros(n+1,1);
% for i=1:m,
%   h = sigmoid(theta'*x(:,i));
%   temp = y(i) - h; 
%   for j=1:n+1,
%     grad(j) = grad(j) + temp * x(j,i); 
%   end;
% end;
% 
% %% Implementation 2 
% grad = zeros(n+1,1);
% for i=1:m,
%   grad = grad + (y(i) - sigmoid(theta'*x(:,i)))* x(:,i);
% end;
% 
% 
% % Slow implementation of matrix-vector multiply
% grad = zeros(n+1,1);
% for i=1:m,
%   grad = grad + b(i) * A(:,i);  % more commonly written A(:,i)*b(i)
% end;
%  
% % Fast implementation of matrix-vector multiply
% grad = A*b;
% 
% 
% %% Implementation 3
% grad = x * (y- sigmoid(theta'*x))';
% 






% =============================================================

end
