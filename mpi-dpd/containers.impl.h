namespace ParticleKernels {
    __global__ void upd_stg2_and_1(bool rbcflag, float2 * _pdata, float * _adata,
				   int nparticles, float dt, float _driving_force, float threshold) {
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

	int type; float driving_force, mass, vx = s1.y, y = s0.y;
	if      (rbcflag                             ) type = MEMB_TYPE;
	else if (!rbcflag &&  lastbit::get(vx)) type =  IN_TYPE;
	else if (!rbcflag && !lastbit::get(vx)) type = OUT_TYPE;
	mass          = (type == MEMB_TYPE) ? rbc_mass : 1;
	/* TODO: no driving force on "inner" particles */
	driving_force = (type ==   IN_TYPE) ? 0        : _driving_force;
	if (doublepoiseuille && y <= threshold) driving_force *= -1;

	s1.y += (ax/mass + driving_force) * dt;
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
	    lastbit::Preserver up1(pdata[laneid].y);
	    pdata[laneid] = s0;
	}

	if (laneid + 32 < nwords) {
	    lastbit::Preserver up1(pdata[laneid + 32].y);
	    pdata[laneid + 32] = s1;
	}

	if (laneid + 64 < nwords) {
	    lastbit::Preserver up1(pdata[laneid + 64].y);
	    pdata[laneid + 64] = s2;
	}
    }

    __global__ void clear_velocity(Particle *p, int n)  {
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	if (pid >= n) return;
	lastbit::Preserver up(p[pid].v[0]);
	for(int c = 0; c < 3; ++c) p[pid].v[c] = 0;
    }
} /* end of ParticleKernels */

namespace Cont {
void  upd_stg2_and_1(Particle* pp, Force* ff, int n,
		     bool rbcflag, float driving_force) {
  if (!n) return;
  ParticleKernels::upd_stg2_and_1<<<(n + 127) / 128, 128, 0>>>
    (rbcflag, (float2 *)pp, (float *)ff, n,
     dt, driving_force, globalextent.y * 0.5 - origin.y);
}

void clear_velocity(Particle* pp, int n) {
  if (n)
    ParticleKernels::clear_velocity<<<(n + 127) / 128, 128 >>>(pp, n);
}

void rbc_init() {
  nc = 0;
  int dims[3];
  MC(MPI_Cart_get(m::cart, 3, dims, periods, coords) );

  rbc::get_triangle_indexing(indices, nt);
  nv = rbc::setup();
}

void _initialize(float *device_pp, float (*transform)[4]) {
  rbc::initialize(device_pp, transform);
}
  
int setup(Particle* pp, const char *path2ic) {
  std::vector<TransformedExtent> allrbcs;
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
  MC(MPI_Bcast(&allrbcs_count, 1, MPI_INT, 0, m::cart));

  allrbcs.resize(allrbcs_count);

  int nfloats_per_entry = sizeof(TransformedExtent) / sizeof(float);

  MC(MPI_Bcast(&allrbcs.front(), nfloats_per_entry * allrbcs_count, MPI_FLOAT, 0, m::cart));

  std::vector<TransformedExtent> good;
  int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

  for(std::vector<TransformedExtent>::iterator it = allrbcs.begin(); it != allrbcs.end(); ++it) {
    bool inside = true;
    for(int c = 0; c < 3; ++c)
      inside &= it->com[c] >= coords[c] * L[c] && it->com[c] < (coords[c] + 1) * L[c];
    if (inside) {
      for(int c = 0; c < 3; ++c)
	it->transform[c][3] -= (coords[c] + 0.5) * L[c];
      good.push_back(*it);
    }
  }

  int gs = good.size();
  for(int i = 0; i < gs; ++i)
    _initialize((float *)(pp + nv * i), good[i].transform);
  return gs; /* number of cells */
}

int rbc_remove(Particle* pp, int nv, int *e, int ne) {
  /* remove RBCs with indexes in `e' */
  bool GO = false, STAY = true;
  int ie, i0, i1;
  std::vector<bool> m(Cont::nc, STAY);
  for (ie = 0; ie < ne; ie++) m[e[ie]] = GO;

  for (i0 = i1 = 0; i0 < nc; i0++)
    if (m[i0] == STAY) {
      CC(cudaMemcpy(pp + nv * i1,
		    pp + nv * i0,
		    sizeof(Particle) * nv,
		    D2D));
      i1++;
    }
  int nstay = i1;
  return nstay;
}

void clear_forces(Force* ff, int n) {
  CC(cudaMemsetAsync(ff, 0, sizeof(Force) * n));
}
static void rbc_dump0(const char *format4ply,
		      int nc, Particle *p, int n, int iddatadump) {
    int ctr = iddatadump;
    char buf[200];
    sprintf(buf, format4ply, ctr);

    if(m::rank == 0)
      mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    ply_dump(buf, indices, nc, nt, p, nv, false);
}

void rbc_dump(Particle* p, int n, int iddatadump) {
  rbc_dump0("ply/rbcs-%05d.ply", n / nv, p, n, iddatadump);
}

}
