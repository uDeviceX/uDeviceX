/*
declaration:
cudaError_t cudaMemcpy (void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)

*/

int i;

int nsol = particles->size, szp = sizeof(Particle);
Particle *sol_dev = particles->xyzuvw.data, *sol_hst = particles_datadump.data;

/* copy from device */
cudaMemcpy(sol_hst, sol_dev, szp*nsol, cudaMemcpyDeviceToHost);

/* set the last bit to 0 for all particles */
#define SUU sol_hst[i].u[0]
for (i = 0; i < nsol; i++) last_bit_float::set(SUU, false);

/* process RBCs */
if (rbcscoll) {
    int nrbc = rbcscoll->pcount();
    Particle *rbc_dev = rbcscoll->xyzuvw.data, *rbc_hst = sol_hst + nsol;

    /* copy from device */
    cudaMemcpy(rbc_hst, rbc_dev, szp*nrbc, cudaMemcpyDeviceToHost);

#define SXX sol_hst[i].x[0]
#define SYY sol_hst[i].x[1]
#define SZZ sol_hst[i].x[2]

#define RXX rbc_hst[i].x[0]
#define RYY rbc_hst[i].x[1]
#define RZZ rbc_hst[i].x[2]

    for (i = 0; i < nsol; i++) {sol_xx[i] = SXX; sol_yy[i] = SYY; sol_zz[i] = SZZ;}
    for (i = 0; i < nrbc; i++) {rbc_xx[i] = RXX; rbc_yy[i] = RYY; rbc_zz[i] = RZZ;}

    iotags_all(nrbc, rbc_xx, rbc_yy, rbc_zz,
               nsol, sol_xx, sol_yy, sol_zz,
               iotags);

    /* set the last bit to 1 for tagged particles */
    for (i = 0; i < nsol; i++) last_bit_float::set(SUU, iotags[i] != -1);
    int cnt; for (i = cnt = 0; i < nsol; i++) if (iotags[i] != -1) cnt++;
    fprintf(stderr, "cnt: %d\n", cnt);

    /* copy to device */
    cudaMemcpy(rbc_dev, rbc_hst, szp*nrbc, cudaMemcpyHostToDevice);
}

/* copy to device */
cudaMemcpy(sol_dev, sol_hst, szp*nsol, cudaMemcpyHostToDevice);
