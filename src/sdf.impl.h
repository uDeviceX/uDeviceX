namespace sdf {
void setup() {
    k_sdf::texSDF.normalized = 0;
    k_sdf::texSDF.filterMode = cudaFilterModePoint;
    k_sdf::texSDF.mipmapFilterMode = cudaFilterModePoint;
    k_sdf::texSDF.addressMode[0] = cudaAddressModeWrap;
    k_sdf::texSDF.addressMode[1] = cudaAddressModeWrap;
    k_sdf::texSDF.addressMode[2] = cudaAddressModeWrap;
}

void init() {
    int N[3]; float extent[3];

    field::ini_dims("sdf.dat", N, extent);
    const int np = N[0] * N[1] * N[2];
    float *grid_data = new float[np];
    float *field     = new float[XTE * YTE * ZTE];
    field::ini_data("sdf.dat", np, grid_data);

    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TE[3] = {XTE, YTE, ZTE};
    MSG0("sampling the geometry file");
    {
	float start[3], spacing[3];
	for (int c = 0; c < 3; ++c) {
	    start[c] = N[c] * (m::coords[c] * L[c] - MARGIN[c]) /
		(float)(m::dims[c] * L[c]);
	    spacing[c] = N[c] * (L[c] + 2 * MARGIN[c]) /
		(float)(m::dims[c] * L[c]) / (float)TE[c];
	}
	float amplitude_rescaling = (XS /*+ 2 * XWM*/) /
	    (extent[0] / m::dims[0]);
	field::sample(start, spacing, TE, N, amplitude_rescaling, grid_data,
		      field);
    }

    if (field_dumps) field::dump(N, extent, grid_data);

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray
       (&arrSDF, &fmt, make_cudaExtent(XTE, YTE, ZTE)));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr
	((void *)field, XTE * sizeof(float), XTE, YTE);

    copyParams.dstArray = arrSDF;
    copyParams.extent = make_cudaExtent(XTE, YTE, ZTE);
    copyParams.kind = H2D;
    CC(cudaMemcpy3D(&copyParams));

    setup();
    CC(cudaBindTextureToArray(k_sdf::texSDF, arrSDF, fmt));

    delete[] grid_data;
    delete[] field;
}

/* sort solvent particle (dev) into remaining in solvent (dev) and turning into wall (hst)*/
static void bulk_wall0(/*io*/ Particle *s_pp, int* s_n, /*o*/ Particle *w_pp, int *w_n,
		       /*w*/ int *keys) {
  int n = *s_n;
  int k, a = 0, b = 0, w = 0; /* all, bulk, wall */
  k_sdf::fill_keys<<<k_cnf(n)>>>(s_pp, n, keys);
  for (/* */ ; a < n; a++) {
    cD2H(&k, &keys[a], 1);
    if      (k == W_BULK) {cD2D(&s_pp[b], &s_pp[a], 1); b++;}
    else if (k == W_WALL) {cD2H(&w_pp[w], &s_pp[a], 1); w++;}
  }
  *s_n = b; *w_n = w;
}

void bulk_wall(/*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n) {
  int *keys;
  mpDeviceMalloc(&keys);
  bulk_wall0(s_pp, s_n, w_pp, w_n, keys);
  CC(cudaFree(keys));
}

void close() {
    CC(cudaUnbindTexture(k_sdf::texSDF));
    CC(cudaFreeArray(arrSDF));
}
}
