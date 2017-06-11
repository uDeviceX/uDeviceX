void ini() {
  m::Comm_dup(::m::cart, &cart);
  sdstr::ini(cart, rank);
  mpDeviceMalloc(&subi_lo); /* 1.5 * numberdensity * XS * YS * ZS */
  mpDeviceMalloc(&subi_re); /* was 1.5*numberdensity*(XS*YS*ZS-(XS-2)*(YS-2)*(ZS-2)) */
  mpDeviceMalloc(&iidx);
  mpDeviceMalloc(&pp_re);
  CC(cudaMalloc(&count_zip, sizeof(count_zip[0])*XS*YS*ZS));
}

void fin() {
  CC(cudaFree(subi_lo));
  CC(cudaFree(subi_re));
  CC(cudaFree(iidx));
  CC(cudaFree(pp_re));
  CC(cudaFree(count_zip));
}

void distr_s(Particle *s_pp, Particle *s_pp0, float4 *s_zip0, ushort4 *s_zip1,
	     int *ps_n, CellLists *cells) {
  int s_n = *ps_n;

  int nbulk, nhalo_padded, nhalo;
  sdstr::post_recv(cart, rank, /**/ recv_size_req, recv_mesg_req);
  if (s_n) {
    sdstr::halo(s_pp, s_n);
    sdstr::scan(s_n);
    sdstr::pack(s_pp, s_n);
  }
  if (!first) {
    sdstr::waitall(send_size_req);
    sdstr::waitall(send_mesg_req);
  }
  first = false;
  nbulk = sdstr::send_sz(cart, rank, send_size_req);
  sdstr::send_msg(cart, rank, send_mesg_req);

  CC(cudaMemsetAsync(cells->count, 0, sizeof(int)*XS*YS*ZS));
  if (s_n)
    k_common::subindex_local<false><<<k_cnf(s_n)>>>
      (s_n, (float2*)s_pp, /*io*/ cells->count, /*o*/ subi_lo);

  sdstr::waitall(recv_size_req);
  sdstr::recv_count(&nhalo_padded, &nhalo);
  sdstr::waitall(recv_mesg_req);
  if (nhalo)
    sdstr::unpack
      (nhalo_padded, /*io*/ cells->count, /*o*/ subi_re, pp_re);

  k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>
    (XS*YS*ZS, (int4*)cells->count, /**/ (uchar4*)count_zip);
  k_scan::scan(count_zip, XS*YS*ZS, /**/ (uint*)cells->start);

  if (s_n)
    k_sdstr::scatter<<<k_cnf(s_n)>>>
      (false, subi_lo,  s_n, cells->start, /**/ iidx);

  if (nhalo)
    k_sdstr::scatter<<<k_cnf(nhalo)>>>
      (true, subi_re, nhalo, cells->start, /**/ iidx);

  s_n = nbulk + nhalo;
  if (s_n)
    k_sdstr::gather<<<k_cnf(s_n)>>>
      ((float2*)s_pp, (float2*)pp_re, s_n, iidx,
       /**/ (float2*)s_pp0, s_zip0, s_zip1);

  
  std::swap(s_pp, s_pp0);
  *ps_n = s_n;
}
