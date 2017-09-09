void distr_rbc() {
    rdstr::extents(r::q.pp, r::q.nc, r::q.nv, /**/ &r::tde);
    rdstr::get_pos(r::q.nc, /**/ &r::tde);
    rdstr::get_reord(r::q.nc, &r::tde, /**/ &r::tdp);
    rdstr::pack(r::q.pp, r::q.nv, &r::tdp, /**/ &r::tds);
    rdstr::post_recv(/**/ &r::tdp, &r::tdc, &r::tdr);
    rdstr::post_send(r::q.nv, &r::tdp, /**/ &r::tdc, &r::tds);
    rdstr::wait_recv(/**/ &r::tdc, &r::tdr);
    r::q.nc = rdstr::unpack(r::q.nv, &r::tdr, &r::tdp, /**/ r::q.pp);
    r::q.n = r::q.nc * r::q.nv;
    rdstr::shift(r::q.nv, &r::tdp, /**/ r::q.pp);

    dSync();
    CC(cudaPeekAtLastError());
}
