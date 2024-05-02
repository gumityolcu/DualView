for file in ../config_files/cluster/explain/$1/*$2.yaml; do
    echo $file
    sbatch explain_job_$1.sh $file
    sleep 1
done;
