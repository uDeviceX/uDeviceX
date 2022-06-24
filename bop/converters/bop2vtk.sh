# !/bin/bash
# 
# Convert multiple .bop files to .vtk files
# run: ./<script_name> <rundirs>

# Loop over different simulation directories
for d; do
    o=$d/vtk
    mkdir -p $o

    files=`ls $d/bop/*.bop`
    
    # Loop over bop files in each sim. dir.
    for f in $files; do
        b=`basename $f`
        bop2vtk ${o}/${b}.vtk ${f}
    done
done
