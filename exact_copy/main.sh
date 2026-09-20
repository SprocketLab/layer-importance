
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <model> <window-to-ssm-expand>" >&2
  exit 2
fi
model=$1
wtos=$2

#STEPS=${STEPS:-15000}
#SEEDS_PER_JOB=${SEEDS_PER_JOB:-1}
#EVAL_EVERY=${EVAL_EVERY:-500}
#EVAL_EXAMPLES=${EVAL_EXAMPLES:-4096}
#MEM=${MEM:-1024}

# export STEPS, SEEDS_PER_JOB, EVAL_EVERY, EVAL_SAMPLES, MEM

for run_number in $(seq 1 5); do
    for dim in 8 12 16 24 32 48 64 96 128; do
		for lr in 1e-3 3e-3 1e-2 3e-2; do
			echo "$run_number $model $dim $lr $wtos"
		done
    done
done | xargs -n 5 -P 6 bash -c 'python3 main.py --token-dims $2 --configs construction:$1 \
  --steps 15000 --seed $0 --lrs $3 \
  --eval-every 500 --eval-examples 4096 \
  --device cuda --memory-budget 1024 --w-to-s $4'
#   --device cpu --max-gpu-gb 0 --memory-budget 1024 '
