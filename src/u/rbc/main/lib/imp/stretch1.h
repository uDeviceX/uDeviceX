void ini(const char* path, int nv, /**/ RbcStretch** fp) {
    rbc_stretch_ini(path, nv, fp);
}

void fin(RbcStretch* f) {
    rbc_stretch_fin(f);
}

void apply(int nm, const RbcStretch* fo, /**/ Force* f) {
    rbc_stretch_apply(nm, fo, f);
}
