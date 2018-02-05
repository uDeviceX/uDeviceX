void time_step_accel_ini(/**/ TimeStepAccel **pq) {
    TimeStepAccel *q;
    EMALLOC(1, &q);
    q->k = 0;
    *pq = q;
}
void time_step_accel_fin(TimeStepAccel *q) { EFREE(q); }
void time_step_accel_push(TimeStepAccel *q, float m, int n, Force *ff) {
    int k;
    k = q->k;
    if (k + 1 == MAX_ACCEL_NUM)
        ERR("k + 1 = %d < %d = MAX_ACCEL_NUM", k + 1, MAX_ACCEL_NUM);
    if (m <= 0) ERR("m <= 0");
    q->mm[k] = m; q->nn[k] = n; q->fff[k] = ff;
    q->k = k + 1;
}

static float accel_max(MPI_Comm comm, TimeStepAccel *q) {
    int n, i;
    float accel, force, mass, max;
    Force *ff;
    max = 0;
    for (i = 0; i < q->k; i++) {
        mass = q->mm[i]; n = q->nn[i]; ff = q->fff[i];
        force = force_stat_max(n, ff);
        accel = force/mass;
        if (accel > max) max = accel;
    }
    UC(max_float(comm, &max));
    return max;
}
