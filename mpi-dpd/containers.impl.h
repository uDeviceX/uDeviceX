namespace ParticleKernels {
    __global__ void upd_stg1(bool rbcflag, Particle * p, Acceleration * a, int n, float dt,
			     float _driving_acceleration, float threshold, bool doublePoiseuille) {
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	if (pid >= n) return;

	int type; float driving_acceleration, mass, vx = p[pid].u[0], y = p[pid].x[1];
	if      ( rbcflag                            ) type = MEMB_TYPE;
	else if (!rbcflag &&  last_bit_float::get(vx)) type =   IN_TYPE;
	else if (!rbcflag && !last_bit_float::get(vx)) type =  OUT_TYPE;
	mass                 = (type == MEMB_TYPE) ? rbc_mass : 1;
	driving_acceleration = (type ==   IN_TYPE) ? 0        : _driving_acceleration;
	if (doublePoiseuille && y <= threshold) driving_acceleration *= -1;

	for(int c = 0; c < 3; ++c) {
	  last_bit_float::Preserver up0(p[pid].u[0]);
	  p[pid].u[c] += (a[pid].a[c]/mass + (c == 0 ? driving_acceleration : 0)) * dt * 0.5;
	}
	for(int c = 0; c < 3; ++c) p[pid].x[c] += p[pid].u[c] * dt;
    }

    __global__ void upd_stg2_and_1(bool rbcflag, float2 * _pdata, float * _adata,
				   int nparticles, float dt, float _driving_acceleration, float threshold,
				   bool doublePoiseuille) {
	int warpid = threadIdx.x >> 5;
	int base = 32 * (warpid + 4 * blockIdx.x);
	int nsrc = min(32, nparticles - base);

	float2 *pdata = _pdata + 3 * base;
	float *adata = _adata + 3 * base;

	int laneid;
	asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

	int nwords = 3 * nsrc;

	float2 s0, s1, s2;
	float ax, ay, az;

	if (laneid < nwords)
	{
	    s0 = __ldg(pdata + laneid);
	    ax = __ldg(adata + laneid);
	}

	if (laneid + 32 < nwords)
	{
	    s1 = __ldg(pdata + laneid + 32);
	    ay = __ldg(adata + laneid + 32);
	}

	if (laneid + 64 < nwords)
	{
	    s2 = __ldg(pdata + laneid + 64);
	    az = __ldg(adata + laneid + 64);
	}

	{
	    int srclane0 = (3 * laneid + 0) & 0x1f;
	    int srclane1 = (srclane0 + 1) & 0x1f;
	    int srclane2 = (srclane0 + 2) & 0x1f;

	    int start = laneid % 3;

	    {
		float t0 = __shfl(start == 0 ? s0.x : start == 1 ? s1.x : s2.x, srclane0);
		float t1 = __shfl(start == 0 ? s2.x : start == 1 ? s0.x : s1.x, srclane1);
		float t2 = __shfl(start == 0 ? s1.x : start == 1 ? s2.x : s0.x, srclane2);

		s0.x = t0;
		s1.x = t1;
		s2.x = t2;
	    }

	    {
		float t0 = __shfl(start == 0 ? s0.y : start == 1 ? s1.y : s2.y, srclane0);
		float t1 = __shfl(start == 0 ? s2.y : start == 1 ? s0.y : s1.y, srclane1);
		float t2 = __shfl(start == 0 ? s1.y : start == 1 ? s2.y : s0.y, srclane2);

		s0.y = t0;
		s1.y = t1;
		s2.y = t2;
	    }

	    {
		float t0 = __shfl(start == 0 ? ax : start == 1 ? ay : az, srclane0);
		float t1 = __shfl(start == 0 ? az : start == 1 ? ax : ay, srclane1);
		float t2 = __shfl(start == 0 ? ay : start == 1 ? az : ax, srclane2);

		ax = t0;
		ay = t1;
		az = t2;
	    }
	}

	int type; float driving_acceleration, mass, vx = s1.y, y = s0.y;
	if      (rbcflag                             ) type = MEMB_TYPE;
	else if (!rbcflag &&  last_bit_float::get(vx)) type =  IN_TYPE;
	else if (!rbcflag && !last_bit_float::get(vx)) type = OUT_TYPE;
	mass                 = (type == MEMB_TYPE) ? rbc_mass : 1;
	driving_acceleration = (type ==   IN_TYPE) ? 0        : _driving_acceleration;
	if (doublePoiseuille && y <= threshold) driving_acceleration *= -1;

	s1.y += (ax/mass + driving_acceleration) * dt;
	s2.x += ay/mass * dt;
	s2.y += az/mass * dt;

	s0.x += s1.y * dt;
	s0.y += s2.x * dt;
	s1.x += s2.y * dt;

	{
	    int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
	    int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
	    int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

	    int start = laneid % 3;

	    {
		float t0 = __shfl(s0.x, srclane0);
		float t1 = __shfl(s2.x, srclane1);
		float t2 = __shfl(s1.x, srclane2);

		s0.x = start == 0 ? t0 : start == 1 ? t2 : t1;
		s1.x = start == 0 ? t1 : start == 1 ? t0 : t2;
		s2.x = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }

	    {
		float t0 = __shfl(s0.y, srclane0);
		float t1 = __shfl(s2.y, srclane1);
		float t2 = __shfl(s1.y, srclane2);

		s0.y = start == 0 ? t0 : start == 1 ? t2 : t1;
		s1.y = start == 0 ? t1 : start == 1 ? t0 : t2;
		s2.y = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }

	    {
		float t0 = __shfl(ax, srclane0);
		float t1 = __shfl(az, srclane1);
		float t2 = __shfl(ay, srclane2);

		ax = start == 0 ? t0 : start == 1 ? t2 : t1;
		ay = start == 0 ? t1 : start == 1 ? t0 : t2;
		az = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }
	}

	if (laneid < nwords) {
	    last_bit_float::Preserver up1(pdata[laneid].y);
	    pdata[laneid] = s0;
	}

	if (laneid + 32 < nwords) {
	    last_bit_float::Preserver up1(pdata[laneid + 32].y);
	    pdata[laneid + 32] = s1;
	}

	if (laneid + 64 < nwords) {
	    last_bit_float::Preserver up1(pdata[laneid + 64].y);
	    pdata[laneid + 64] = s2;
	}
    }

    __global__ void clear_velocity(Particle *p, int n)  {
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	if (pid >= n) return;
	last_bit_float::Preserver up(p[pid].u[0]);
	for(int c = 0; c < 3; ++c) p[pid].u[c] = 0;
    }
} /* end of ParticleKernels */

namespace Cont {
void upd_stg1(ParticleArray* pa, bool rbcflag, float driving_acceleration, cudaStream_t stream) {
  if (pa->pp.S)
    ParticleKernels::upd_stg1<<<(pa->pp.S + 127) / 128, 128, 0, stream>>>
      (rbcflag, pa->pp.D, pa->aa.D, pa->pp.S,
       dt, driving_acceleration, globalextent.y * 0.5 - origin.y, doublepoiseuille);
}

void  upd_stg2_and_1(ParticleArray* pa, bool rbcflag, float driving_acceleration, cudaStream_t stream) {
  if (pa->pp.S)
    ParticleKernels::upd_stg2_and_1<<<(pa->pp.S + 127) / 128, 128, 0, stream>>>
      (rbcflag, (float2 *)pa->pp.D, (float *)pa->aa.D, pa->pp.S,
       dt, driving_acceleration, globalextent.y * 0.5 - origin.y, doublepoiseuille);
}

void clear_velocity(ParticleArray* pa) {
  if (pa->pp.S)
    ParticleKernels::clear_velocity<<<(pa->pp.S + 127) / 128, 128 >>>(pa->pp.D, pa->pp.S);
}

#define     resize2(b1, b2, n) b1.resize(n), (b2).resize(n)
#define rbc_resize2(b1, b2, n) Cont::ncells = (n), resize2(b1, b2, Cont::nvertices*(n))

void rbc_init() {
  ncells = 0;
  int dims[3], periods[3];
  MC( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

  CudaRBC::get_triangle_indexing(indices, ntriangles);
  CudaRBC::Extent extent;
  CudaRBC::setup(nvertices, extent);
}

void _initialize(float *device_pp, float (*transform)[4]) {
  CudaRBC::initialize(device_pp, transform);
}

void setup(ParticleArray* pa, const char *path2ic) {
    vector<TransformedExtent> allrbcs;
    if (rank == 0) {
	//read transformed extent from file
	FILE *f = fopen(path2ic, "r");
	printf("READING FROM: <%s>\n", path2ic);
	bool isgood = true;
	while(isgood) {
	    float tmp[19];
	    for(int c = 0; c < 19; ++c) {
		int retval = fscanf(f, "%f", tmp + c);
		isgood &= retval == 1;
	    }

	    if (isgood) {
		TransformedExtent t;
		for(int c = 0; c < 3; ++c) t.com[c] = tmp[c];

		int ctr = 3;
		for(int c = 0; c < 16; ++c, ++ctr) t.transform[c / 4][c % 4] = tmp[ctr];
		allrbcs.push_back(t);
	    }
	}
	fclose(f);
	printf("Instantiating %d CELLs from...<%s>\n", (int)allrbcs.size(), path2ic);
    } /* end of myrank == 0 */

    int allrbcs_count = allrbcs.size();
    MC(MPI_Bcast(&allrbcs_count, 1, MPI_INT, 0, cartcomm));

    allrbcs.resize(allrbcs_count);

    int nfloats_per_entry = sizeof(TransformedExtent) / sizeof(float);

    MC(MPI_Bcast(&allrbcs.front(), nfloats_per_entry * allrbcs_count, MPI_FLOAT, 0, cartcomm));

    vector<TransformedExtent> good;
    int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

    for(vector<TransformedExtent>::iterator it = allrbcs.begin(); it != allrbcs.end(); ++it) {
	bool inside = true;
	for(int c = 0; c < 3; ++c)
	    inside &= it->com[c] >= coords[c] * L[c] && it->com[c] < (coords[c] + 1) * L[c];
	if (inside) {
	    for(int c = 0; c < 3; ++c)
		it->transform[c][3] -= (coords[c] + 0.5) * L[c];
	    good.push_back(*it);
	}
    }

    rbc_resize2(pa->pp, pa->aa, good.size());
    for(int i = 0; i < good.size(); ++i)
      _initialize((float *)(pa->pp.D + nvertices * i), good[i].transform);
}

void remove(ParticleArray* pa, int *entries, int nentries) {
    std::vector<bool > marks(ncells, true);
    std::vector< int > survivors;

    for(int i = 0; i < nentries; ++i) marks[entries[i]] = false;
    for(int i = 0; i < ncells  ; ++i) if (marks[i]) survivors.push_back(i);

    int nsurvived = survivors.size();
    DeviceBuffer<Particle> survived(nvertices*nsurvived);
    for(int i = 0; i < nsurvived; ++i)
	CC(cudaMemcpy(survived.D + nvertices * i, pa->pp.D + nvertices * survivors[i],
		      sizeof(Particle) * nvertices, cudaMemcpyDeviceToDevice));

    rbc_resize2(pa->pp, pa->aa, nsurvived);
    CC(cudaMemcpy(pa->pp.D, survived.D,
		  sizeof(Particle) * survived.S, cudaMemcpyDeviceToDevice));
}

int  pcount() {return ncells * nvertices;}
void clear_acc(ParticleArray* pa, cudaStream_t stream) {CC(cudaMemsetAsync(pa->aa.D, 0, sizeof(Acceleration) * pa->aa.S, stream));}
static void rbc_dump0(const char *format4ply,
		      MPI_Comm comm, int ncells,
		      Particle *p, Acceleration *a, int n, int iddatadump) {
    int ctr = iddatadump;

    //we fused VV stages so we need to recover the state before stage 1
    for(int i = 0; i < n; ++i) {
	last_bit_float::Preserver up(p[i].u[0]);
	for(int c = 0; c < 3; ++c) {
	    p[i].x[c] -= dt * p[i].u[c];
	    p[i].u[c] -= 0.5 * dt * a[i].a[c];
	}
    }
    char buf[200];
    sprintf(buf, format4ply, ctr);

    int rank;
    MC(MPI_Comm_rank(comm, &rank));

    if(rank == 0)
      mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    ply_dump(comm, cartcomm, buf, indices, ncells, ntriangles, p, nvertices, false);
}

void rbc_dump(MPI_Comm comm, Particle* p, Acceleration* a, int n, int iddatadump) {
  rbc_dump0("ply/rbcs-%05d.ply", comm, n / nvertices, p, a, n, iddatadump);
}

}
