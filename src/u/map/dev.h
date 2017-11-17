__global__ void main() {
    using namespace map;

    int dx, dy, dz;
    dx = dy = dz = -1;

    do {
        printf("%d %d %d\n", dx, dy, dz);
    } while (nxt_xyz0(&dx, &dy, &dz) != END);

}
