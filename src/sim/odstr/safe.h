void odstr() {
    dbg::check_pos_pu(o::q.pp, o::q.n, F("B"));
    sub::odstr();
    dbg::check_pos_pu(o::q.pp, o::q.n, F("A"));
}
