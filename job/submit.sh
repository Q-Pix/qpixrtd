#!/usr/bin/env bash

# MUST RUN FROM ./job DIRECTORY

bin=/n/home02/jh/repos/qpixrtd/EXAMPLE/build/EXAMPLE

# NOTE:
# i don't have write access to /n/holystore01/LABS/guenette_lab/Lab/data/QPIX
# and its subdirectories
#   -- jh
in_dir=/n/holystore01/LABS/guenette_lab/Lab/data/QPIX/Supernova_Test
out_dir=/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova_test/rtd

script_dir=./scripts
log_dir=./log

# for path in `find $in_dir -maxdepth 1 -mindepth 1 -type d`
for path in $(find "$in_dir" -maxdepth 1 -mindepth 1 -type d)
do

    subdir=$(basename "$path")
    seed_arr=()
    script=run_${subdir}.sh
    prefix=""

    for file in `find $path -maxdepth 1 -mindepth 1 -type f`
    do
        if [[ -f $file ]]; then
            filename=$(basename -- "$file")
            extension="${filename##*.}"
            filename="${filename%.*}"
            arr=(${filename//-/ })
            prefix=${arr[0]}
            seed=${arr[1]}
            seed_arr+=($seed)
        fi
    done

    printf -v joined '%s,' "${seed_arr[@]}"
    # seeds="${joined%,}"

cat > ${script_dir}/${script} << EOL
#!/usr/bin/env bash

#SBATCH -J qpix_supernova  # A single job name for the array
#SBATCH -n 1               # Number of cores
#SBATCH -N 1               # All cores on one machine
#SBATCH -p guenette        # Partition
#SBATCH --mem 1000         # Memory request (Mb)
#SBATCH -t 0-1:00          # Maximum execution time (D-HH:MM)
#SBATCH -o ${log_dir}/${subdir}_%A_%a.out  # Standard output
#SBATCH -e ${log_dir}/${subdir}_%A_%a.err  # Standard error

seed_arr=(${seed_arr[@]})
seed=\${seed_arr[\${SLURM_ARRAY_TASK_ID}]}

${bin} ${path}/${prefix}-\${seed}.root ${out_dir}/${subdir}/${prefix}-rtd-\${seed}.root
EOL

    n=${#seed_arr[@]}

    if [ "$n" -gt 0 ];
    then
        cmd="sbatch --array=0-$(( $n - 1 )) ${script_dir}/${script}"
        echo $cmd
        $cmd
    fi

done
