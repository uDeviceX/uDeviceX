/* set tags of particles according to the RBCs */
void gen_tags() {
    collision::get_tags(o::q.pp, o::q.n, r::tt.texvert, r::tt.textri, r::q.nt, r::q.nv, r::q.nc, /**/ o::qt.ii);
}
