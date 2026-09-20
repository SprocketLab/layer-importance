TASK_NAME="segmented-retrieval"
HEADS=1
STATE_DIM=1
NUM_VOCAB=20
NUM_NUMBERS=20
SEQ_LEN=150
MIN_SEQ_LEN=99
WINDOW=20
NUM_EXAMPLES=4000
MEMORY=6000
export TASK_NAME HEADS STATE_DIM NUM_VOCAB NUM_NUMBERS SEQ_LEN MIN_SEQ_LEN WINDOW NUM_EXAMPLES MEMORY

for run_number in $(seq 1 5); do
    #for model in TTTT STTT SSTT SSST SSSS TSSS STSS SSTS TSTT TTST TTTS; do
	#for model in TSTT TSST STST TTST TSTS SSTT TTSS SSSS TTTT STTS; do
	#for model in TTSTT TSSST STSSS SSSTS STTTS; do
    #for model in TTT TTS TST TSS STT STS SST SSS; do
	for model in TTTT TTTS TTST TTSS TSTT TSTS TSST TSSS STTT STTS STST STSS SSTT SSTS SSST SSSS; do
	#for model in TTTTT TTTTS TTTST TTTSS TTSTT TTSTS TTSST TTSSS TSTTT TSTTS TSTST TSTSS TSSTT TSSTS TSSST TSSSS STTTT STTTS STTST STTSS STSTT STSTS STSST STSSS SSTTT SSTTS SSTST SSTSS SSSTT SSSTS SSSST SSSSS; do
    #for model in TT TS ST SS; do
        #for hidden_size in 64; do
        for hidden_size in 96; do
        #for hidden_size in 32 48 64 96 128; do
			for window in 70; do
			#for window in 5 10 20 30 50 100; do
				for state_dim in 1; do
					echo "$run_number $model $hidden_size $window $state_dim"
				done
			done
        done
    done
done | xargs -n 5 -P 6 bash -c 'python3 main.py --run_number $1 --train_task $TASK_NAME --eval_task $TASK_NAME --model $2 --hidden_size $3 --window $4 --heads $HEADS --state_dim $5 --save_results True --auto_lr True --num_vocab $NUM_VOCAB --num_numbers $NUM_NUMBERS --sequence_length $SEQ_LEN --eval_sequence_length $SEQ_LEN --epochs 10 --num_examples $NUM_EXAMPLES --min_train_length $MIN_SEQ_LEN --max_train_length $SEQ_LEN --min_eval_length $MIN_SEQ_LEN --max_eval_length $SEQ_LEN --balance_mem $MEMORY' _
