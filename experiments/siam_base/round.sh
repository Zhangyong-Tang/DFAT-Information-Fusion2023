show_help() {
cat << EOF
Usage: 
    ${0##*/} [-h/--help] [-s/--start] [-e/--end] [-d/--dataset] [-m/--model]  [-g/--gpu]
    e.g.
        bash ${0##*/} -s 1 -e 20 -d VOT2018 -g 4 # for test models
        bash ${0##*/} -m snapshot/checkpoint_e10.pth -n 8 -g 4 # for tune models
EOF
}

ROOT=`git rev-parse --show-toplevel`

dataset=VOT2018
NUM=4
START=9
END=10
GPU=4
j=0

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit
            ;;
        -d|--dataset)
            dataset=$2
            shift 2
            ;;
        -n|--num)
            NUM=$2
            shift 2
            ;;
        -s|--start)
            START=$2
            shift 2
            ;;
        -e|--end)
            END=$2
            shift 2
            ;;
        -m|--model)
            model=$2
            shift 2
            ;;
        -g|--gpu)
            GPU=$2
            shift 2
            ;;
        *)
            echo invalid arg [$1]
            show_help
            exit 1
            ;;
    esac
done

set -e

if [ -z "$model" ]; then
    echo test snapshot $START ~ $END on dataset $dataset with $GPU gpus.
    for i in $(seq $START $END)
    do 
        #bash test.sh snapshot/checkpoint_e$i.pth $dataset $(($i % $GPU)) &
        CUDA_VISIBLE_DEVICES=$((($i + $j) % $GPU)) python -u $ROOT/tools/test.py \
    --config config.yaml \
    --snapshot snapshot/checkpoint_refine_e$i.pth \
    --dataset $dataset &
    done
    wait

    python $ROOT/tools/eval.py --dataset $dataset --num 9 2>&1 | tee logs/eval_test_$dataset.log
else
    echo tuning $model on dataset $dataset with $NUM jobs in $GPU gpus.
    for i in $(seq 1 $NUM)
    do 
        bash tune.sh $model $dataset $(($i % $GPU)) & 
    done
    wait
    rm finish.flag

    python $ROOT/tools/eval.py --dataset $dataset --num 9 2>&1 | tee logs/eval_tune_$dataset.log
fi
