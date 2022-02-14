clc; clear; close all;

X = [338, 333, 328, 207, 226, 25, 179, 60, 208, 606];
Y = [640, 633, 619, 393, 428, 27, 193, 66, 226, 1591];
L =@(w,b) MSE(w, b, X, Y);


%% Gradient Descent
figure(1)
Pos = zeros(1000000,2);
theta0 = [-4, -120]; theta = theta0;
lr = 1e-6;
for t=1:1000000
    [theta, criterion] = GD(X, Y, lr, theta);
    Pos(t,:) = theta;
    
    if criterion==1
        break
    end
end

subplot(1,3,1)
PlotErrorSurf(Pos(1:t,:), L)
title(['GD iteration: ' num2str(t)])

%% Adagrad
Pos = zeros(1000000,2);
theta0 = [-4, -120]; theta = theta0;
lr = 1; g = 0;
for t=1:1000000
    [theta, g, criterion] = Adagrad(X, Y, lr, theta, g);
    Pos(t,:) = theta;
    
    if criterion==1
        break
    end
end

subplot(1,3,2)
PlotErrorSurf(Pos(1:t,:), L)
title(['Adagrad iteration: ' num2str(t)])


%% Adam
clc;
Pos = zeros(100000,2);
theta0 = [-4, -120]; theta = theta0;
lr = 0.01; m = zeros(size(theta)); v = zeros(size(theta));
for t=1:100000
    [theta, m, v, criterion, L_theta] = Adam(X, Y, lr, theta, m, v, t);
    Pos(t,:) = theta;
    
    if criterion==1
        L_theta
        break
    end
end

subplot(1,3,3)
PlotErrorSurf(Pos(1:t,:), L)
title(['Adam iteration: ' num2str(t)])



