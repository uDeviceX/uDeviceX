# !/bin/bash
#
# run as: me.sh <dir-list>

for f; do
    echo $f

    cfiles=(`ls $f/bop/colors_solvent-*.values | sort -g`)
    dfiles=(`ls $f/bop/solvent-*.values | sort -g`)
    nf=`ls $f/bop/solvent-*.values | wc -l`

    for i in `seq 1 $nf`; do
        j=$(($i-1))
        echo $j
        cf=${cfiles[${j}]}
        df=${dfiles[${j}]}
        python color2xyz.py $cf $df
    done
done
