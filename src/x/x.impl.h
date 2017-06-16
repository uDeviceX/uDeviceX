void init() {
  m::Comm_dup(::m::cart, &cart);
  odstr::ini(cart, rank);
  mpDeviceMalloc(&subi_lo);
  mpDeviceMalloc(&subi_re);
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
  odstr::post_recv(cart, rank, /**/ recv_size_req, recv_mesg_req);
  if (n) {
    odstr::halo(pp, n);
    odstr::scan(n);
    odstr::pack(pp, n);
  }
  if (!first) {
    odstr::waitall(send_size_req);
    odstr::waitall(send_mesg_req);
  }
  first = false;
  nbulk = odstr::send_sz(cart, rank, send_size_req);
  odstr::send_msg(cart, rank, send_mesg_req);

  CC(cudaMemsetAsync(cells->count, 0, sizeof(int)*XS*YS*ZS));
  if (n)
    k_common::subindex_local<false><<<k_cnf(n)>>>
      (n, (float2*)pp, /*io*/ cells->count, /*o*/ subi_lo);

  odstr::waitall(recv_size_req);
  odstr::recv_count(&nhalo_padded, &nhalo);
  odstr::waitall(recv_mesg_req);
  if (nhalo)
    odstr::unpack
      (nhalo_padded, /*io*/ cells->count, /*o*/ subi_re, pp_re);

  k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>
    (XS*YS*ZS, (int4*)cells->count, /**/ (uchar4*)count_zip);
  k_scan::scan(count_zip, XS*YS*ZS, /**/ (uint*)cells->start);

  if (n)
    k_odstr::scatter<<<k_cnf(n)>>>
      (false, subi_lo,  n, cells->start, /**/ iidx);

  if (nhalo)
    k_odstr::scatter<<<k_cnf(nhalo)>>>
      (true, subi_re, nhalo, cells->start, /**/ iidx);

  n = nbulk + nhalo;
  if (n)
    k_odstr::gather<<<k_cnf(n)>>>
      ((float2*)pp, (float2*)pp_re, n, iidx,
       /**/ (float2*)pp0, zip0, zip1);
  *pn = n;
}
