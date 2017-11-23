void run_eq(long te) { /* equilibrate */
    scheme::force::Param fpar = {
        .a = 0,
        .b = 0,
        .c = 0
    };
    
    bool wall0 = false;
    for (long it = 0; it < te; ++it) step(&fpar, wall0, it);
    distribute_flu();
}

void run(long ts, long te) {
    long it; /* current timestep */
    dump_strt_templ(); /* :TODO: is it the right place? */

    scheme::force::Param fpar = {
        .a = FORCE_PAR_A,
        .b = 0,
        .c = 0
    };
    
    /* ts, te: time start and end */
    for (it = ts; it < te; ++it) {
        glb::step(it - ts, te - ts, dt); /* set kernel globals */
        step(&fpar, walls, it);
    }
    distribute_flu();
}
