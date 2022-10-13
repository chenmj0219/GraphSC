#### GIN fine-tuning



for dataset in bbbp sider clintox toxcast bace tox21; do
    for epochs in 20 40 60 80 100 ; do
        for runseed in 0 1 2 3 4 5 6 7 8 9; do
            CUDA_VISIBLE_DEVICES=1 python finetune.py --input_model_file models_mgsc/mgsc_80.pth \
            --split scaffold --runseed $runseed --gnn_type gin --dataset $dataset --lr 0.005 --epochs $epochs
done
done
done