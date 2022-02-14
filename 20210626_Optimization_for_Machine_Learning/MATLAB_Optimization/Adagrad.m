function [theta, g, criterion] = Adagrad(X, Y, lr, theta, g)
L_theta = UpdateGrad(X, Y, theta);
g = g + L_theta.^2;
theta = theta- (lr./sqrt(g)).*L_theta;

if sum(abs(L_theta) < 1e-5) == 1
    criterion = 1;
else
    criterion = 0;
end

end