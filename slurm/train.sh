for file in ../config_files/cluster/train/$1/*.yaml; do
    echo $file
    sbatch train_job_$1.sh $file
    sleep 1
done;
