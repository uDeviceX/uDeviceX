void run_eq(long te) { /* equilibrate */
  long it;
  float driving_force0 = 0;
  bool wall0 = false;
  for (it = 0; it < te; ++it) step(driving_force0, wall0, it);
}

void run(long ts, long te) {
  /* ts, te: time start and end */
  long it;
  float driving_force0 = pushflow ? driving_force : 0;
  for (it = ts; it < te; ++it) step(driving_force0, walls, it);
}
