void ini0(cudaArray *arrsdf, dev::tex3Dca<float> *texsdf,
	  int N[3], float extent[3], float* grid_data) {
    float sc;

    float *field     = new float[XTE * YTE * ZTE];
    MC(l::m::Barrier(m::cart));

    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TE[3] = {XTE, YTE, ZTE};
    float start[3], spacing[3];
    for (int c = 0; c < 3; ++c) {
      start[c] = N[c] * (m::coords[c] * L[c] - MARGIN[c]) /
	(float)(m::dims[c] * L[c]);
      spacing[c] = N[c] * (L[c] + 2 * MARGIN[c]) /
	(float)(m::dims[c] * L[c]) / (float)TE[c];
    }
    field::sample(start, spacing, TE, N, grid_data, /**/ field);

    sc = XS / (extent[0] / m::dims[0]);
    field::scale(TE, sc, /**/ field);

    if (field_dumps) field::dump(N, extent, grid_data);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr
        ((void *)field, XTE * sizeof(float), XTE, YTE);

    copyParams.dstArray = arrsdf;
    copyParams.extent = make_cudaExtent(XTE, YTE, ZTE);
    copyParams.kind = H2D;
    CC(cudaMemcpy3D(&copyParams));

    texsdf->setup(arrsdf);

    delete[] field;
}

void ini(cudaArray *arrsdf, dev::tex3Dca<float> *texsdf) {
  enum {X, Y, Z};
  int N[3];
  float ext[3]; /* extent */
  int n;
  float *D; /* data */

  field::ini_dims("sdf.dat", N, ext);
  n = N[X] * N[Y] * N[Z];
  D = new float[n];
  field::ini_data("sdf.dat", n, D);
  ini0(/*o*/ arrsdf, texsdf, /*i*/ N, ext, D);
  delete [] D;
}
/* sort solvent particle (dev) into remaining in solvent (dev) and turning into wall (hst)*/
static void bulk_wall0(const dev::tex3Dca<float> texsdf, /*io*/ Particle *s_pp, int* s_n,
                       /*o*/ Particle *w_pp, int *w_n, /*w*/ int *keys) {
    int n = *s_n;
    int k, a = 0, b = 0, w = 0; /* all, bulk, wall */
    dev::fill_keys<<<k_cnf(n)>>>(texsdf, s_pp, n, keys);
    for (/* */ ; a < n; a++) {
        cD2H(&k, &keys[a], 1);
        if      (k == W_BULK) {cD2D(&s_pp[b], &s_pp[a], 1); b++;}
        else if (k == W_WALL) {cD2H(&w_pp[w], &s_pp[a], 1); w++;}
    }
    *s_n = b; *w_n = w;
}

void bulk_wall(const dev::tex3Dca<float> texsdf, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n) {
    int *keys;
    mpDeviceMalloc(&keys);
    bulk_wall0(texsdf, s_pp, s_n, w_pp, w_n, keys);
    CC(cudaFree(keys));
}

/* bulk predicate : is in bulk? */
static bool bulkp(int *keys, int i) {
    int k; cD2H(&k, &keys[i], 1);
    return k == W_BULK;
}

static int who_stays0(int *keys, int nc, int nv, /*o*/ int *stay) {
    int c, v;  /* cell and vertex */
    int s = 0; /* how many stays? */
    for (c = 0; c < nc; ++c) {
        v = 0;
        while (v  < nv && bulkp(keys, v + nv * c)) v++;
        if    (v == nv) stay[s++] = c;
    }
    return s;
}

static int who_stays1(const dev::tex3Dca<float> texsdf, Particle *pp, int n, int nc, int nv, /**/ int *stay, /*w*/ int *keys) {
    dev::fill_keys<<<k_cnf(n)>>>(texsdf, pp, n, keys);
    return who_stays0(keys, nc, nv, /*o*/ stay);
}

int who_stays(const dev::tex3Dca<float> texsdf, Particle *pp, int n, int nc, int nv, /**/ int *stay) {
    int *keys;
    CC(cudaMalloc(&keys, n*sizeof(keys[0])));
    nc = who_stays1(texsdf, pp, n, nc, nv, /**/ stay, /*w*/ keys);
    CC(cudaFree(keys));
    return nc;
}
