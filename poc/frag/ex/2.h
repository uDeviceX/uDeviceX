#define a(x) (assert(x))
const int *d; /* direction */

d = frag_to_dir[frag_bulk];
a(d[X]  ==  0);  a(d[Y] ==  0);  a(d[Z] ==  0);
