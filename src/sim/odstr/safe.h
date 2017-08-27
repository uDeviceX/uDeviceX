void odstr() {
    dbg::check_pp_pu(o::q.pp, o::q.n, F("B"));
    sub::odstr();
    dbg::check_pp_pu(o::q.pp, o::q.n, F("A"));
}
