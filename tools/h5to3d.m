#!/usr/bin/env octave-qf

% convert h5part file to several 3D files

% Usage (octave-qf wrapper should be on the PATH):
% > ./h5to3d.m test_data/all.h5part 3d

# TEST: h5to3d.t1
# rm -rf 3d
# ./h5to3d.m test_data/all.h5part 3d
# cat 3d/000[01].3D  > h5to3d.out.txt

fn = argv(){1}; % file name
% fn = 'test_data/all.h5part'; % input file

d = argv(){2};
% d  = '3d'                   ; % output directory

fmt = '%s/%04d.3D'          ; % format of output file
D   = load('-hdf5', fn);

% number of timesteps
nt = length(fieldnames(D));

gs = @(ts)    getfield(D, sprintf('Step_%d', ts)); % get string
gf = @(ts, f) getfield(gs(ts), f); % get field
fi = @(x, idx) x(idx); % filter by idx
ou = @(ts)     sprintf(fmt, d, ts);

mkdir(d)
for ts=0:nt-1 % loop over timesteps
  xx = gf(ts, 'x'); yy = gf(ts, 'y'); zz = gf(ts, 'z'); type = gf(ts, 'type');
  idx = (type == 1); % extract solid
  xx = fi(xx, idx); yy = fi(yy, idx); zz = fi(zz, idx);
  dlmwrite(ou(ts), [xx', yy', zz'], ' ');
  fprintf(stderr(), '(h5to3d.m) writing %s\n', ou(ts))
end


