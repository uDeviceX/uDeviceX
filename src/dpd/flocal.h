namespace dpd {
void flocal(float4 *zip0, ushort4 *zip1, int n,
	    int *start, int *count,
	    l::rnd::d::KISS* rnd, /**/
	    Force *ff) {
    if (n > 0)
      flocal0(zip0, zip1, (float*)ff, n,
	      start, count, 1, XS, YS, ZS, rnd->get_float());
}
} /* namespace dpd */
