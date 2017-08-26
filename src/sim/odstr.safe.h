void odstr() {
    dbg::check_pp_pu(o::q.pp, o::q.n, "odstr.safe: befor");
    sub::odstr();
    dbg::check_pp_pu(o::q.pp, o::q.n, "odstr.safe: after");
}
