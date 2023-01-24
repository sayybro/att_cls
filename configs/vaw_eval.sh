python main.py \
    --pretrained checkpoints/vaw/checkpoint.pth \
    --att_det \
    --dataset_file vaw \
    --data_path data/vaw \
    --num_obj_classes 80 \
    --backbone resnet50 \
    --eval