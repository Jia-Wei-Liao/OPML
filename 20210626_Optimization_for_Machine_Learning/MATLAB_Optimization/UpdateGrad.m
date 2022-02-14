function L_theta = UpdateGrad(X, Y, theta)
tmp = -2*(Y - theta(1)*X - theta(2)*ones(size(X)));
Lw = dot(tmp, X); Lb = sum(tmp);
L_theta = [Lw, Lb];

end