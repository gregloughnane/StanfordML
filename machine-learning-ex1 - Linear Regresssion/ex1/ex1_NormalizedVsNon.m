%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise()

%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data2.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
% Plot Data
% Note: You have to complete the code in plotData.m
subplot(2,1,1)
plot(X, y, 'rx', 'MarkerSize', 10);
xlabel('Profit fin $10,000s')
ylabel('Population of City in 10,000s')
%% =================== Part 3: Cost and Gradient descent ===================

% THIS FOR NORMALIZED
[X_norm mu sigma] = featureNormalize(X); %Changes the answer
X_norm = [ones(m, 1), X_norm]; % Add a column of ones to x
subplot(2,1,2)
plot(X_norm(:,2), y, 'rx', 'MarkerSize', 10);
xlabel('Normalized Profit')
ylabel('Population of City in 10,000s')
X = [ones(m, 1), X]; % THIS FOR NON-NORMALIZED

%% Solve Least Squares Problem to check with Steepest Descent Convergence
beta_hat = pinv(X'*X)*X'*y
% %Values from Fitted Model & Error Calculation
% Y_hat = X*beta_hat
% error = y - Y_hat
% %Error Variance estimate
% No_ind_vars = 2;
% SigmaSq_hat = (y'*y - beta_hat'*X'*y) / (size(y,1)-No_ind_vars)

beta_hat_norm = pinv(X_norm'*X_norm)*X_norm'*y
% %Values from Fitted Model & Error Calculation
% Y_hat = X_norm*beta_hat
% error = y - Y_hat
% %Error Variance estimate
% No_ind_vars = 2;
% SigmaSq_hat = (y'*y - beta_hat'*X_norm'*y) / (size(y,1)-No_ind_vars)
%%-----------------------------------------------------------

theta = zeros(2, 1); % initialize fitting parameters
thetaOrig = theta;

% Some gradient descent settings
iterations = 15000;
alpha = 1;

% run gradient descent
tic
[theta, J_history, thetaVals] = gradientDescent(X, y, theta, alpha, iterations);
toc

% print theta to screen
fprintf('Theta found by gradient descent (ORIGINAL):\n');
fprintf('%f\n', theta);

% run gradient descent
tic
[theta_norm, J_history_norm, thetaVals_norm] = gradientDescent(X_norm, y, theta, alpha, iterations);
toc

% print theta to screen
fprintf('Theta found by gradient descent (NORMALIZED):\n');
fprintf('%f\n', theta_norm);

% Plot the linear fit
subplot(2,1,1)
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off

subplot(2,1,2)
hold on
plot(X_norm(:,2), X_norm*theta_norm, '-')
legend('Normalized Training data', 'Normalized Linear regression')
hold off % don't overlay any more plots on this figure

  
%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

    
    %% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

    
% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals', logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
% hold on;
% plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
% 
% %Draw Arrows on the Contour Plot to show Progress
% for i = 0:1:iterations-1
%     if i == 0
%         dp = thetaVals(1,:)-thetaOrig';    
%         quiver(thetaOrig(1),thetaOrig(2),dp(1),dp(2));
%         text(thetaOrig(1),thetaOrig(2), sprintf('(%0.5f,%0.5f)',thetaOrig))
%     else
%         dp = thetaVals(i+1,:) - thetaVals(i,:);    
%         quiver(thetaVals(i,1),thetaVals(i,2),dp(1),dp(2));
%     end
% %     if i > 0 && i <3
% %         text(thetaVals(i,1),thetaVals(i,2), sprintf('(%0.3f,%0.3f)',thetaVals(i,:)))
% %     end
% end
% 
% hold off

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');


%% Plot Cost Function vs. Iterations
figure
plot(J_history)
hold on
plot(J_history_norm)
xlabel('Iteration')
ylabel('Cost Function J(theta)')
axis([0 iterations 0 10]);
legend('No Feature Scaling', 'Feature Scaling')








