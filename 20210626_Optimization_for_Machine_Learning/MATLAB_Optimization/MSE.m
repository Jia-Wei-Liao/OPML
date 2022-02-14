function loss = MSE(w, b, X, Y)
loss = 0;
for i = 1:length(X)
    loss = loss + (Y(i) - w.*X(i)-b).^2;
end

end