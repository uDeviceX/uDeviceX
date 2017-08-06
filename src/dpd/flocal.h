namespace dpd {
void flocal(float4 *zip0, ushort4 *zip1, int n,
	    int *start, int *count,
	    rnd::KISS* rnd, /**/
	    Force *ff) {
    if (n > 0)
      flocal0(zip0, zip1, (float*)ff, n,
	      start, count, XS, YS, ZS, rnd->get_float());
}
}
