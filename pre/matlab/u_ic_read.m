function M = u_ic_read(path)
  % u_ic_read  read initial conditions file
  %   M = u_ic_read(path)
  %
  %   each row M(row, :) has the form
  %   [x, y, z,   px, py, pz]
  % where
  % x,y,z is a center of mass postions
  % px,py,pz are rotation angles in degree
  %
  % Examples:
  %
  % M = u_ic_read(path)
  %
  X = 1; Y = 2; Z = 3;
  f = fopen(path, 'r');
  if f == -1
    error('not a file "%s"', path);
  end
  [M, cnt] = fscanf(f, '%d %d %d  %d %d %d', [6, Inf]);
  M = M';
  
  if (cnt == 0) | (mod(cnt, 6) ~= 0)
    error('fail to read "%s"', path)
  endif
  
  if fclose(f) ~= 0
    error('fail to close "%s"', path);
  end
end
