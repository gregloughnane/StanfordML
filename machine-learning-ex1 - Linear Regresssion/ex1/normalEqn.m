function [beta_hat] = normalEqn(X, Y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

beta_hat = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------


% Normal Equation Solution (Use when < 10,000 sq matrix)
%Regression Coefficients
beta_hat = pinv(X'*X)*X'*Y;

% -------------------------------------------------------------


% ============================================================

end
