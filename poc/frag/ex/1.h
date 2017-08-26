#define a(x) (assert(x))
const int *to_d, *fro_d; /* [to, from] directions */
int fid, x = -1, y = 0, z = 1;

fid = frag_to_id(x, y, z);
to_d  = frag_to_dir[fid];
fro_d = frag_fro_dir[fid];
a(to_d[X]  ==  x);  a(to_d[Y] ==  y);  a(to_d[Z] ==  z);
a(fro_d[X] == -x); a(fro_d[Y] == -y); a(fro_d[Z] == -z);
