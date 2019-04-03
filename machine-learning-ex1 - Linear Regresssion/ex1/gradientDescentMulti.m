function [theta, J_history, thetaVals] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
thetaLen = length(theta);


tempVal = theta; %Temporary Value to store theta 

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


     % Store new predicted values of theta
    temp = (X*theta - y);
    
    % Store new predicted values of theta
    for i=1:thetaLen
        tempVal(i,1) = sum(temp.*X(:,i));
    end
    
    theta = theta - (alpha/m)*tempVal;
    
    
    
    %Store theta0 and theta1 values for plotting
     thetaVals(iter,:) = theta;
     

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

