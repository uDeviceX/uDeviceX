namespace tclist {

/* maximum number of triangles per solute 
   this is used to encode the solute id */
#define MAXT (256*256*256) 

static __device__ int encode(int soluteid, int id) {
    return soluteid * MAXT + id;
}



} // tclist
