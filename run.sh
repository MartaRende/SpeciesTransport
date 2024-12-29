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

echo "Copying directory"
# copy project in destination path
name=$(date +'%H%M%S')

ssh marta.rende@192.168.91.13 -t "mkdir -p /home/marta.rende/speciesTransport/$name/output "
echo "here1"
rsync -az --progress ./solve/ marta.rende@192.168.91.13:/home/marta.rende/speciesTransport/$name/solve &
rsync -az --progress ./write/ marta.rende@192.168.91.13:/home/marta.rende/speciesTransport/$name/write &
rsync -az --progress ./initialization/ marta.rende@192.168.91.13:/home/marta.rende/speciesTransport/$name/initialization &
#rsync -az --progress ./unitTests/ marta.rende@192.168.91.13:/home/marta.rende/speciesTransport/$name/unitTests &
#rsync --progress ./common_includes.c marta.rende@192.168.91.13:/home/marta.rende/speciesTransport/$name &
rsync --progress ./main.cpp marta.rende@192.168.91.13:/home/marta.rende/speciesTransport/$name 
rsync --progress ./Makefile marta.rende@192.168.91.13:/home/marta.rende/speciesTransport/$name 

echo "Starting compilation"
ssh marta.rende@192.168.91.13 "$(sshqfunc remote)" -- $name