#define a(x) (assert(x))
int ad0, ad1, id, x = -1, y = 0, z = 1;
id = frag_to_id(x, y, z);
ad0 = frag_anti(id);

ad1 = frag_to_id(-x, -y, -z);
assert(ad0 == ad1);
