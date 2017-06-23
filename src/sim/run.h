void run_eq(long te) { /* equilibrate */
    float driving_force0 = 0;
    bool wall0 = false;
    for (long it = 0; it < te; ++it) step(driving_force0, wall0, it);
}

void run(long ts, long te) {
    dump_strt_templ(); /* :TODO: is it the right place? */
    /* ts, te: time start and end */
    float driving_force0 = pushflow ? driving_force : 0;
    for (long it = ts; it < te; ++it) step(driving_force0, walls, it);
}
