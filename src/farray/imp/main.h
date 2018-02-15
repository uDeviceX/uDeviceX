void farray_push_pp(Force *ff, FoArray *a) {
    a->ff = (float*) ff;
    a->stress = false;
}

void farray_push_ss(float *ss, FoArray *a) {
    a->ss = ss;
    a->stress = true;
}

bool farray_has_stress(const FoArray *a) {
    return a->stress;
}

void farray_get_view(const FoArray *a, FoArray_v *v) {
    v->ff = a->ff;
}

void farray_get_view(const FoArray *a, FoSArray_v *v) {
    v->ff = a->ff;
    v->ss = a->ss;
}
