struct MeshBB {
    int *ncols;       /* number of possible collisions per particle      */
    float4 *datacol;  /* list of data related to collisions per particle */
    int *idcol;       /* list of triangle colliding ids per particle     */
};
