# map

A structure to walk over particles of a `fragment`.

API:

	void map::ini(Frag, float x, float y, float z, /**/ map::M*);
	int map::fst(/**/ map::M);
	int map::nxt(int i, /**/ map::M);
	int map::endp(int i, map::M)

`fst` returns a first particle, `nxt` returns a next particle, `endp`
returns `0` if `i` is a last particle
