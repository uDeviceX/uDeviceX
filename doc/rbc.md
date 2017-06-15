# hdr

	texture<float2, 1, cudaReadModeElementType> texVertices;
	texture<int,    1, cudaReadModeElementType> texAdjVert;
	texture<int,    1, cudaReadModeElementType> texAdjVert2;
	texture<int4,            cudaTextureType1D> texTriangles4;

# imp

	void setup(int* faces)
	void forces(int nc, Particle *pp, Force *ff, float* host_av)
	int setup(Particle* pp, int nv, /*w*/ Particle *pp_hst)

from containers.impl.h

	rbc_dump(int nc, Particle *p, int* triplets, int nv, int nt, int id) {

# function called by sim

* `rbc::forces(r::nc, r::pp, r::ff, r::av);`
* `rbc::setup(r::faces);`

# varibles from sim

    int n = 0, nc = 0, nt = RBCnt, nv = RBCnv;
    Particle *pp;
    Force    *ff;

    Particle pp_hst[MAX_PART_NUM];
    int faces[MAX_FACE_NUM];
    float *av;
