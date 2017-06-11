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

void distr_s(Particle *s_pp, Particle *s_pp0, float4 *s_zip0, ushort4 *s_zip1,
	     int *ps_n, CellLists *cells) {
  int s_n = *ps_n;

  int nbulk, nhalo_padded, nhalo;
  sdstr::post_recv(s_cart, s_rank, /**/ s_recv_size_req, s_recv_mesg_req);
  if (s_n) {
    sdstr::halo(s_pp, s_n);
    sdstr::scan(s_n);
    sdstr::pack(s_pp, s_n);
  }
  if (!s_first) {
    sdstr::waitall(s_send_size_req);
    sdstr::waitall(s_send_mesg_req);
  }
  s_first = false;
  nbulk = sdstr::send_sz(s_cart, s_rank, s_send_size_req);
  sdstr::send_msg(s_cart, s_rank, s_send_mesg_req);

  CC(cudaMemsetAsync(cells->count, 0, sizeof(int)*XS*YS*ZS));
  if (s_n)
    k_common::subindex_local<false><<<k_cnf(s_n)>>>
      (s_n, (float2*)s_pp, /*io*/ cells->count, /*o*/ s_subi_lo);

  sdstr::waitall(s_recv_size_req);
  sdstr::recv_count(&nhalo_padded, &nhalo);
  sdstr::waitall(s_recv_mesg_req);
  if (nhalo)
    sdstr::unpack
      (nhalo_padded, /*io*/ cells->count, /*o*/ s_subi_re, s_pp_re);

  k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>
    (XS*YS*ZS, (int4*)cells->count, /**/ (uchar4*)s_count_zip);
  k_scan::scan(s_count_zip, XS*YS*ZS, /**/ (uint*)cells->start);

  if (s_n)
    k_sdstr::scatter<<<k_cnf(s_n)>>>
      (false, s_subi_lo,  s_n, cells->start, /**/ s_iidx);

  if (nhalo)
    k_sdstr::scatter<<<k_cnf(nhalo)>>>
      (true, s_subi_re, nhalo, cells->start, /**/ s_iidx);

  s_n = nbulk + nhalo;
  if (s_n)
    k_sdstr::gather<<<k_cnf(s_n)>>>
      ((float2*)s_pp, (float2*)s_pp_re, s_n, s_iidx,
       /**/ (float2*)s_pp0, s_zip0, s_zip1);

  
  std::swap(s_pp, s_pp0);
  *ps_n = s_n;
}
