TASK_NAME="tst-task"
HEADS=1
NUM_VOCAB=20
NUM_NUMBERS=20
SEQ_LEN=150
MIN_SEQ_LEN=99
NUM_EXAMPLES=4000
MEMORY=512
export TASK_NAME HEADS NUM_VOCAB NUM_NUMBERS SEQ_LEN MIN_SEQ_LEN NUM_EXAMPLES MEMORY

for run_number in $(seq 1 5); do
    for model in TS ST; do
		OLDIFS=$IFS; IFS=',';
		#for i in 128,1,1 64,1,1 32,1,1, 16,1,1 8,1,1 4,1,1 64,3,1 32,3,1 16,3,1 8,3,1 4,3,1 32,7,1 16,7,1 8,7,1 4,7,1 16,15,1 8,15,1 4,15,1 8,31,1 4,31,1 256,1,1 128,3,1 64,7,1 32,15,1 16,31,1 8,63,1; do
		#for i in 128,1,1 64,1,1 32,1,1, 16,1,1 8,1,1 4,1,1 64,3,1 32,3,1 16,3,1 8,3,1 4,3,1 32,7,1 16,7,1 8,7,1 4,7,1 16,15,1 8,15,1 4,15,1 8,31,1 4,31,1; do
		for i in 192,1,1 96,1,1 48,1,1 24,1,1 12,1,1 6,1,1 96,3,1 48,3,1 24,3,1 12,3,1 6,3,1 48,7,1 24,7,1 12,7,1 6,7,1 24,15,1 12,15,1 6,15,1 12,31,1 6,31,1 6,63,1; do
		#for i in 96,1,1 48,1,1 24,1,1 12,1,1 6,1,1 48,3,1 24,3,1 12,3,1 6,3,1 24,7,1 12,7,1 6,7,1 12,15,1 6,15,1; do
			set -- $i
			echo "$run_number $model $1 $2 $3"
        done
		IFS=$OLDIFS
    done
done | xargs -n 5 -P 6 bash -c 'python3 main.py --run_number $1 --train_task $TASK_NAME --eval_task $TASK_NAME --model $2 --hidden_size $3 --window $4 --heads $HEADS --state_dim $5 --save_results True --auto_lr True --num_vocab $NUM_VOCAB --num_numbers $NUM_NUMBERS --sequence_length $SEQ_LEN --eval_sequence_length $SEQ_LEN --epochs 10 --num_examples $NUM_EXAMPLES --min_train_length $MIN_SEQ_LEN --max_train_length $SEQ_LEN --min_eval_length $MIN_SEQ_LEN --max_eval_length $SEQ_LEN --balance_mem $MEMORY' _

for run_number in $(seq 1 5); do
    for model in TT; do
		OLDIFS=$IFS; IFS=',';
		#for i in 256,1,0 128,1,0 64,2,0 32,4,0 16,8,0 8,16,0 4,32,0; do
		#for i in 128,1,0 64,2,0 32,4,0 16,8,0 8,16,0 4,32,0; do
		for i in 192,1,0 96,1,0 48,2,0 24,4,0 12,8,0 6,16,0; do
		#for i in 96,1,0 48,2,0 24,4,0 12,8,0 6,16,0; do
			set -- $i
			echo "$run_number $model $1 $2 $3"
        done
		IFS=$OLDIFS
    done
done | xargs -n 5 -P 6 bash -c 'python3 main.py --run_number $1 --train_task $TASK_NAME --eval_task $TASK_NAME --model $2 --hidden_size $3 --window $4 --heads $HEADS --state_dim $5 --save_results True --auto_lr True --num_vocab $NUM_VOCAB --num_numbers $NUM_NUMBERS --sequence_length $SEQ_LEN --eval_sequence_length $SEQ_LEN --epochs 10 --num_examples $NUM_EXAMPLES --min_train_length $MIN_SEQ_LEN --max_train_length $SEQ_LEN --min_eval_length $MIN_SEQ_LEN --max_eval_length $SEQ_LEN --balance_mem $MEMORY' _

for run_number in $(seq 1 5); do
    for model in SS; do
		OLDIFS=$IFS; IFS=',';
		#for i in 256,0,1 128,0,1 64,0,2 32,0,4 16,0,8 8,0,16 4,0,32; do
		#for i in 128,0,1 64,0,2 32,0,4 16,0,8 8,0,16 4,0,32; do
		for i in 192,0,1 96,0,2 48,0,4 24,0,8 12,0,16 6,0,32; do
		#for i in 96,0,2 48,0,4 24,0,8 12,0,16 6,0,32; do
			set -- $i
			echo "$run_number $model $1 $2 $3"
        done
		IFS=$OLDIFS
    done
done | xargs -n 5 -P 6 bash -c 'python3 main.py --run_number $1 --train_task $TASK_NAME --eval_task $TASK_NAME --model $2 --hidden_size $3 --window $4 --heads $HEADS --state_dim $5 --save_results True --auto_lr True --num_vocab $NUM_VOCAB --num_numbers $NUM_NUMBERS --sequence_length $SEQ_LEN --eval_sequence_length $SEQ_LEN --epochs 10 --num_examples $NUM_EXAMPLES --min_train_length $MIN_SEQ_LEN --max_train_length $SEQ_LEN --min_eval_length $MIN_SEQ_LEN --max_eval_length $SEQ_LEN --balance_mem $MEMORY' _
