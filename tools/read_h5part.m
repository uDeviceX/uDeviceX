#!/usr/bin/env octave-qf

% Usage (octave-qf wrapper should be on the PATH):
% > ./read_h5part.m test_data/all.h5part

fn  = argv(){1}; % file name
% fn  = 'test_data/all.h5part';
D   = load('-hdf5', fn);

% ts: timestep; f: field (x, y, ...)
ts = 4; gs = getfield(D, sprintf('Step_%d', ts));
gf = @(f) getfield(gs, f);

xx = gf("x"); yy = gf("y"); zz = gf("z"); tt = gf("type");

idx = tt==1;
dlmwrite(stdout(), [xx(idx)' yy(idx)' zz(idx)'], ' ');
