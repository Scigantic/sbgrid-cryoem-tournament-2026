#!/bin/bash
# Copy a running job's newest RELION iteration (optimiser, model, sampling, data, half maps)
# from its node-local scratch to shared storage every 20 minutes, so a timeout or a lost node
# costs one iteration, not the run. Runs on the login node; the copy itself runs on the job's
# node (srun --overlap). Stops when the job is no longer running.
#   setsid nohup ~/pipeline/ckpt_copy.sh 64 /scratch/merge/Refine8 ~/results/merge/Refine8 \
#     > ~/ckpt64.log 2>&1 < /dev/null &
job=$1 src=$2 dst=$3 last=
mkdir -p "$dst"
while squeue -h -j "$job" -t R 2>/dev/null | grep -q .; do
  it=$(srun --jobid="$job" --overlap -n1 bash -c "ls -t $src/run_it*_optimiser.star | head -1" 2>/dev/null)
  if [[ -n "$it" && "$it" != "$last" ]]; then
    if srun --jobid="$job" --overlap -n1 bash -c "rsync -a ${it%_optimiser.star}_* $dst/" 2>/dev/null; then
      last=$it
      echo "$(date +%m-%d\ %H:%M) copied $(basename "$it")"
    fi
  fi
  sleep 1200
done
echo "$(date +%m-%d\ %H:%M) job $job no longer running"
