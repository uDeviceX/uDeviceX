namespace dump
{
    void init()
    {
        if (m::rank == 0)
        mkdir("bop", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    
    void parts(const Particle *pp, const long n, const char *name, const int step)
    {
        char fname[256] = {0};
        sprintf(fname, "bop/%s-%05d.bop", name, step);

        MPI_File f;
        MPI_Status status;
        MPI_Offset base, offset = 0;
        MPI_Offset len = n * sizeof(Particle);
        
        MC( MPI_File_open(m::cart, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f) );

        MC( MPI_File_set_size(f, 0) );
        MC( MPI_File_get_position(f, &base) ); 

        if (m::rank == 0)
        {
            MC( MPI_File_write (f, &n, 1, MPI_LONG, &status));
            len += sizeof(long);
            base += sizeof(long);
        }

        MC( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart) );
        
        MC( MPI_File_write_at_all(f, base + offset, pp, n, Particle::datatype(), &status) ); 
        
        MC( MPI_File_close(&f) );
    }

}
