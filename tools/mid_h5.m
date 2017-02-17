#!/usr/bin/env octave-qf

% Usage (octave-qf wrapper should be on the PATH):
% > ./mid_h5.m test_data/ff.h5

fn  = argv(){1}; % file name
X = 1; Y = 2; Z = 3;
sq  = @squeeze;
grd = @(n) (1:n) - 1/2;

load(fn, 'u', 'v', 'w');
vx = sq(u); vy = sq(v); vz = sq(w);
nx = size(vx, X); ny = size(vy, Y); nz = size(vz, Z);
xx = grd(nx); yy = grd(ny); zz = grd(nz);

lo_x = nx/2 - 2; hi_x = nx/2 + 2;
vx = vx(lo_x:hi_x, :, :); % cut a part in the middle

vx = sq(mean(vx, 1));
vx = sq(mean(vx, 1));

eps = 1e-3; idx = vx < eps; vx(idx) = 0; % filter small

dlmwrite(stdout, vx', '\n');
