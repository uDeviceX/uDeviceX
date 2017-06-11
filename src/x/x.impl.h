void ini() {
  m::Comm_dup(::m::cart, &s_cart);
  sdstr::ini(s_cart, s_rank);
  mpDeviceMalloc(&s_subi_lo); /* 1.5 * numberdensity * XS * YS * ZS */
  mpDeviceMalloc(&s_subi_re); /* was 1.5*numberdensity*(XS*YS*ZS-(XS-2)*(YS-2)*(ZS-2)) */
  mpDeviceMalloc(&s_iidx);
  mpDeviceMalloc(&s_pp_re);
  CC(cudaMalloc(&s_count_zip, sizeof(s_count_zip[0])*XS*YS*ZS));
}

void fin() {
  CC(cudaFree(s_subi_lo));
  CC(cudaFree(s_subi_re));
  CC(cudaFree(s_iidx));
  CC(cudaFree(s_pp_re));
  CC(cudaFree(s_count_zip));
}
