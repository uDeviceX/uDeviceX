#define a(x) (assert(x))
const int *to_d;

to_d = frag_to_dir[frag_bulk];
a(to_d[X]  ==  0);  a(to_d[Y] ==  0);  a(to_d[Z] ==  0);
