#define a(x) (assert(x))
int x = -1, y = 0, z = 1;
a(frag_to_id(x, y, z) == frag_fro_id(-x, -y, -z));
a(frag_to_id(0, 0, 0) == frag_fro_id(0, 0, 0));
