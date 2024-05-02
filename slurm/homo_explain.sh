for file in ../config_files/cluster/explain/$1/homo*.yaml; do
    echo $file
    sbatch explain_job.sh $file
    sleep 1
done;
