function PlotErrorSurf(Pos, L)
dw = 1; db = 10; ww = -10:dw:10; bb = -200:db:-100;
[W, B] = meshgrid(ww, bb);
Error = L(W, B);

surf(W, B, Error, 'FaceAlpha', 0.5); hold on
plot3(Pos(:,1), Pos(:,2), L(Pos(:,1), Pos(:,2)),'r.');
plot3(2.67, -188.4, L(2.67, -188.4),'r*');

end