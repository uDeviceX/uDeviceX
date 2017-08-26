#define a(x) (assert(x))
int ncell;
ncell = frag_ncell(frag_bulk);
a(ncell == XS * YS * ZS );

int id, x = -1, y = 0, z = 1;
id = frag_to_id(x, y, z);
ncell = frag_ncell(id);
assert(ncell == YS);
