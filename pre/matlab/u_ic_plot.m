function [tt, xx, yy, zz] = u_ic_plot(M, t, x, y, z)
  X = 1; Y = 2; Z = 3; PX = 4; PY = 5; PZ = 6;
  nv = numel(x);
  nt = size(t, 1);
  n = size(M, 1); % number of objects

  xx = zeros(nv*n, 1);
  yy = zeros(nv*n, 1);
  zz = zeros(nv*n, 1);
  tt = zeros(nt*n, 3);

  % tri
  from = 1:nt;
  for i=1:n
    to = from + nt*i;
    tt(to, :) = t(from, :) + nv*i;
  end

  % rotation
  for i=1:n
    px = M(i, PX); py = M(i, PY); pz = M(i, PZ);
    R = u_rotmatrix(px, py, pz);
    for j=1:nv
      from = j;
      to   = j + nv*i;
      r    = [x(from); y(from); z(from)];
      r    = R * r;
      xx(to) = r(X); yy(to) = r(Y); zz(to) = r(Z);
    end
  end

  % shift
  from = 1:nv;
  for i=1:n
    to = from + nv*i;
    xx(to) = xx(to) + M(i, X);
    yy(to) = yy(to) + M(i, Y);
    zz(to) = zz(to) + M(i, Z);
  end
end
