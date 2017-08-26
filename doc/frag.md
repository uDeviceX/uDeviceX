# fragment

From `fid` to direction and back again.

    #define a(x) (assert(x))
    const int *to_d, *fro_d; /* [to, from] directions */
    int fid, x = -1, y = 0, z = 1;
    
    fid = frag_to_id(x, y, z);
    to_d  = frag_to_dir[fid];
    fro_d = frag_fro_dir[fid];
    a(to_d[X]  ==  x);  a(to_d[Y] ==  y);  a(to_d[Z] ==  z);
    a(fro_d[X] == -x); a(fro_d[Y] == -y); a(fro_d[Z] == -z);

`frag_bulk` is `fid` of bulk fragment

    #define a(x) (assert(x))
    const int *d; /* direction */
    
    d = frag_to_dir[frag_bulk];
    a(d[X]  ==  0);  a(d[Y] ==  0);  a(d[Z] ==  0);
	
From `fid` to number of cells
	
    #define a(x) (assert(x))
    int ncell;
    ncell = frag_ncell(frag_bulk);
    a(ncell == XS * YS * ZS );
    
    int id, x = -1, y = 0, z = 1;
    id = frag_to_id(x, y, z);
    ncell = frag_ncell(id);
    assert(ncell == YS);

Matching `to_id` and `fro_id`
	
    #define a(x) (assert(x))
    int ncell;
    ncell = frag_ncell(frag_bulk);
    a(ncell == XS * YS * ZS );
    
    int id, x = -1, y = 0, z = 1;
    id = frag_to_id(x, y, z);
    ncell = frag_ncell(id);
    assert(ncell == YS);

