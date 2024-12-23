#!/bin/bash
sshqfunc() { echo "bash -c $(printf "%q" "$(declare -f "$@"); $1 \"\$@\"")"; };
remote() {
    # compile project
    cd /home/marta.rende/speciesTransport/$1
    make

    #run executable file
    #using sbatch
    #sbatch launch-job.sh
    #squeue
    #echo "Waiting for job to finish.. 10s countdown"
    #sleep 10
    #cat output.txt
    ./run_speciesTransport 
}

# copy project in destination path
name=$(date +'%H%M%S')
scp -r "$(pwd)" marta.rende@192.168.91.13:/home/marta.rende/speciesTransport/$name
ssh marta.rende@192.168.91.13 "$(sshqfunc remote)" -- $name