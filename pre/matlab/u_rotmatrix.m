function R = u_rotmatrix(px, py, pz)
  d2r = @(d) d*pi/180;
  px = d2r(px); py = d2r(py); pz = d2r(pz);
  RX = [1        0        0;
        0  cos(px) -sin(px);
	0  sin(px)  cos(px)];
  RY = [cos(py) 0  sin(py);
	0 1        0;
	-sin(py) 0  cos(py)];
  RZ = [cos(pz) -sin(pz) 0;
	sin(pz)  cos(pz)  0;
	0              0  1];
  R = RX * RY * RZ;
end
