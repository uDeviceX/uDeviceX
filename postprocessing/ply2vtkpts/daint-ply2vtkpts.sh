module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
#module unload cray-mpich/7.0.4
#module load cray-mpich/7.1.1
#module unload gcc
#module load gcc/4.9.1
#module load vtk

convert_some()
{
    MYFOLDER=$1
    SRCPATH=$2
    SRCPATTERN=$3
    NVERTPERCELLS=$4

    mkdir -p $MYFOLDER

    echo `date` "convert_some: $*" >> ${MYFOLDER}/log.txt

    find "$SRCPATH" -name "$SRCPATTERN" > /tmp/asd.txt

    for F in $(cat /tmp/asd.txt)
    do
	SRC=`basename $F`

	DST=${MYFOLDER}/${SRC%.ply}.vtp

	aprun ./ply2vtk $NVERTPERCELLS $F $DST
    done
}

if (( $# != 4))
then
    echo "usage ./convert-all.sh <dstfolder> <srcfolder> <nvertices-per-rbc> <nvertices-per-ctc>"

    exit 1
fi


DSTFOLDER=$1 #for example "ichip31"
SRCFOLDER=$2 
NVERTRBC=$3 #for example 498
NVERTCTC=$4 #for example 5220

convert_some "$DSTFOLDER" "$SRCFOLDER" "rbcs-*.ply" $NVERTRBC
convert_some "$DSTFOLDER" "$SRCFOLDER" "ctcs-*.ply" $NVERTCTC