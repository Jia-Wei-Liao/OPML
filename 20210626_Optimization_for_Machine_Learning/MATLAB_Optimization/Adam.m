function [theta, m, v, criterion, L_theta] = Adam(X, Y, lr, theta, m, v, t)
beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
old_theta = theta;
L_theta = UpdateGrad(X, Y, theta);
m = beta1*m + (1-beta1)*L_theta;
v = beta2*v + (1-beta2)*(L_theta.^2);
m_ = m./(1-beta1^t);
v_ = v./(1-beta2^t);
theta = theta - lr*m_./(sqrt(v_)+epsilon);

if abs(theta-old_theta) < 1e-5
    criterion = 1;

else
     criterion = 0;
end

end