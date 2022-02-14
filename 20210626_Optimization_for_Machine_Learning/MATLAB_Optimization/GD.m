function [theta, criterion] = GD(X, Y, lr, theta)
L_theta = UpdateGrad(X, Y, theta);
theta = theta - lr*L_theta;

if sum(abs(L_theta) < 1e-5) == 1
    criterion = 1;
else
    criterion = 0;
end

end