void init() {
  m::Comm_dup(::m::cart, &cart);
  sdstr::ini(cart, rank);
  mpDeviceMalloc(&subi_lo); /* 1.5 * numberdensity * XS * YS * ZS */
  mpDeviceMalloc(&subi_re); /* was 1.5*numberdensity*(XS*YS*ZS-(XS-2)*(YS-2)*(ZS-2)) */
  mpDeviceMalloc(&iidx);
  mpDeviceMalloc(&pp_re);
  CC(cudaMalloc(&count_zip, sizeof(count_zip[0])*XS*YS*ZS));
}

void close() {
  CC(cudaFree(subi_lo));
  CC(cudaFree(subi_re));
  CC(cudaFree(iidx));
  CC(cudaFree(pp_re));
  CC(cudaFree(count_zip));
}

void distr(Particle *pp, Particle *pp0, float4 *zip0, ushort4 *zip1,
	   int *pn, Clist *cells) {
  int n = *pn;

  int nbulk, nhalo_padded, nhalo;
  sdstr::post_recv(cart, rank, /**/ recv_size_req, recv_mesg_req);
  if (n) {
    sdstr::halo(pp, n);
    sdstr::scan(n);
    sdstr::pack(pp, n);
  }
  if (!first) {
    sdstr::waitall(send_size_req);
    sdstr::waitall(send_mesg_req);
  }
  first = false;
  nbulk = sdstr::send_sz(cart, rank, send_size_req);
  sdstr::send_msg(cart, rank, send_mesg_req);

  CC(cudaMemsetAsync(cells->count, 0, sizeof(int)*XS*YS*ZS));
  if (n)
    k_common::subindex_local<false><<<k_cnf(n)>>>
      (n, (float2*)pp, /*io*/ cells->count, /*o*/ subi_lo);

  sdstr::waitall(recv_size_req);
  sdstr::recv_count(&nhalo_padded, &nhalo);
  sdstr::waitall(recv_mesg_req);
  if (nhalo)
    sdstr::unpack
      (nhalo_padded, /*io*/ cells->count, /*o*/ subi_re, pp_re);

  k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>
    (XS*YS*ZS, (int4*)cells->count, /**/ (uchar4*)count_zip);
  k_scan::scan(count_zip, XS*YS*ZS, /**/ (uint*)cells->start);

  if (n)
    k_sdstr::scatter<<<k_cnf(n)>>>
      (false, subi_lo,  n, cells->start, /**/ iidx);

  if (nhalo)
    k_sdstr::scatter<<<k_cnf(nhalo)>>>
      (true, subi_re, nhalo, cells->start, /**/ iidx);

  n = nbulk + nhalo;
  if (n)
    k_sdstr::gather<<<k_cnf(n)>>>
      ((float2*)pp, (float2*)pp_re, n, iidx,
       /**/ (float2*)pp0, zip0, zip1);

  
  std::swap(pp, pp0);
  *pn = n;
}
