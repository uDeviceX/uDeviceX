static bool is_stress_free(const RbcForce *f) {
    return f->stype == RBC_STRESS_FREE;

static void get_stress_view(const RbcForce *f, /**/ StressFul_v *v) {
    *v = f->sinfo.sful;
}

static void get_stress_view(const RbcForce *f, /**/ StressFree_v *v) {
    *v = f->sinfo.sfree;
}

