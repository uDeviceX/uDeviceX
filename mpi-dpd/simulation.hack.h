/*
declaration:
cudaError_t cudaMemcpy (void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)

*/

int nsol = particles->S, szp = sizeof(Particle);
Particle *sol_dev = particles->pp.D, *sol_hst = particles_datadump->data;

/* copy from device */
cudaMemcpy(sol_hst, sol_dev, szp*nsol, cudaMemcpyDeviceToHost);

/* process RBCs */
if (rbcscoll) {
    int nrbc = Cont::pcount();
    Particle *rbc_dev = rbcscoll->pp.D, *rbc_hst = sol_hst + nsol;

    /* copy from device */
    cudaMemcpy(rbc_hst, rbc_dev, szp*nrbc, cudaMemcpyDeviceToHost);

#define SXX sol_hst[i].x[0]
#define SYY sol_hst[i].x[1]
#define SZZ sol_hst[i].x[2]

#define RXX rbc_hst[i].x[0]
#define RYY rbc_hst[i].x[1]
#define RZZ rbc_hst[i].x[2]

#define SUU sol_hst[i].u[0]
    int i;
    for (i = 0; i < nsol; i++) {sol_xx[i] = SXX; sol_yy[i] = SYY; sol_zz[i] = SZZ;}
    for (i = 0; i < nrbc; i++) {rbc_xx[i] = RXX; rbc_yy[i] = RYY; rbc_zz[i] = RZZ;}

    iotags_all(nrbc, rbc_xx, rbc_yy, rbc_zz,
               nsol, sol_xx, sol_yy, sol_zz,
               iotags);

    /* collect statistics */
    int in2out = 0, out2in = 0, cnt_in = 0;
    for (i = 0; i < nsol; i++) {
      bool was_in = last_bit_float::get(SUU), was_out = !was_in;
      bool now_in = iotags[i] != -1         , now_out = !now_in;

      if (was_in  && now_out) in2out ++;
      if (was_out && now_in ) out2in ++;
      if (now_in            ) cnt_in ++;
    }
    /* set the last bit to 1 for tagged particles */
    for (i = 0; i < nsol; i++) last_bit_float::set(SUU, iotags[i] != -1);
    fprintf(stderr, "(simulation.hack.h) in2out, out2in, cnt_in: %d  %d  %d\n",
	    in2out, out2in, cnt_in);

    /* copy to device */
    cudaMemcpy(rbc_dev, rbc_hst, szp*nrbc, cudaMemcpyHostToDevice);
 } else {
  /* set the last bit to 0 for all particles */
  for (int i = 0; i < nsol; i++) last_bit_float::set(SUU, false);
 }

/* copy to device */
cudaMemcpy(sol_dev, sol_hst, szp*nsol, cudaMemcpyHostToDevice);
