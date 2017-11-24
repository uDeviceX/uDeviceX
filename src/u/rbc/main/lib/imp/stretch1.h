void ini(const char* path, int nv, /**/ rbc::stretch::Fo** fp) {
    rbc::stretch::ini(path, nv, fp);
    
}
void fin(rbc::stretch::Fo* f) {
    rbc::stretch::fin(f);
}

void apply(int nm, const rbc::stretch::Fo* fo, /**/ Force* f) {
    rbc::stretch::apply(nm, fo, f);
}
