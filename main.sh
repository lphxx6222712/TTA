
#python train_source.py --device Cirrus
#python train_source.py --device Spectralis
#python train_source.py --device Topcon

#python train_UDA.py --device Topcon
#python train_UDA.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar
#python train_UDA.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar
#
#python train_UDA.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#python train_UDA.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#
#python train_UDA.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#python train_UDA.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar

#python train_UDA_minent.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar
#python train_UDA_minent.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar
##
#python train_UDA_minent.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#python train_UDA_minent.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
##
#python train_UDA_minent.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#python train_UDA_minent.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#
#
#python train_adversial_UDA.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar
#python train_adversial_UDA.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar
###
#python train_adversial_UDA.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#python train_adversial_UDA.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
###
#python train_adversial_UDA.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#python train_adversial_UDA.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar

#CUDA_VISIBLE_DEVICES=0 python train_plain_ADVENT.py --batch-size 6 --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=0 python train_plain_ADVENT.py --batch-size 6 --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar &
#
###
#CUDA_VISIBLE_DEVICES=1 python train_plain_ADVENT.py --batch-size 6 --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=1 python train_plain_ADVENT.py --batch-size 6 --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
###
#####
#CUDA_VISIBLE_DEVICES=2 python train_plain_ADVENT.py --batch-size 14 --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=3 python train_plain_ADVENT.py --batch-size 14 --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &

#CUDA_VISIBLE_DEVICES=0 python SCAN_ST.py --batch-size 6 --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=0 python SCAN_ST.py --batch-size 6 --device Topcon --mode  l-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar &
#
#CUDA_VISIBLE_DEVICES=1 python SCAN_ST.py --batch-size 6 --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=1 python SCAN_ST.py --batch-size 6 --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#
#CUDA_VISIBLE_DEVICES=2 python SCAN_ST.py --batch-size 8 --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=3 python SCAN_ST.py --batch-size 8 --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &


#CUDA_VISIBLE_DEVICES=0 python Adv_ST.py --batch-size 6 --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=0 python Adv_ST.py --batch-size 6 --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar &
#
#CUDA_VISIBLE_DEVICES=1 python Adv_ST.py --batch-size 6 --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=1 python Adv_ST.py --batch-size 6 --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#
#CUDA_VISIBLE_DEVICES=2 python Adv_ST.py --batch-size 12 --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=3 python Adv_ST.py --batch-size 12 --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &

CUDA_VISIBLE_DEVICES=0 python SCAN_ST.py --batch-size 4 --device Topcon --model-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/unet_Cirrus_to_Topcon_Adv_1.26v2/checkpoint/model_best.pth.tar &
CUDA_VISIBLE_DEVICES=0 python SCAN_ST.py --batch-size 4 --device Topcon --model-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/unet_Spectralis_to_Topcon_Adv_1.26v2/checkpoint/model_best.pth.tar &

CUDA_VISIBLE_DEVICES=0 python SCAN_ST.py --batch-size 4 --device Spectralis --model-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/unet_Cirrus_to_Spectralis_Adv_1.26v2/checkpoint/model_best.pth.tar &
CUDA_VISIBLE_DEVICES=1 python SCAN_ST.py --batch-size 4 --device Spectralis --model-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/unet_Topcon_to_Spectralis_Adv_1.26v2/checkpoint/model_best.pth.tar &

CUDA_VISIBLE_DEVICES=1 python SCAN_ST.py --batch-size 4 --device Cirrus --model-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/unet_Spectralis_to_Cirrus_Adv_1.26v2/checkpoint/model_best.pth.tar &
CUDA_VISIBLE_DEVICES=1 python SCAN_ST.py --batch-size 4 --device Cirrus --model-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/unet_Topcon_to_Cirrus_Adv_1.26v2/checkpoint/model_best.pth.tar &

#CUDA_VISIBLE_DEVICES=0 python train_adversial_UDA_with_seperated_layer_head_new.py --batch-size 8 --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=0 python train_adversial_UDA_with_seperated_layer_head_new.py --batch-size 8 --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar &

####
#CUDA_VISIBLE_DEVICES=1 python train_adversial_UDA_with_seperated_layer_head_new.py --batch-size 8 --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=1 python train_adversial_UDA_with_seperated_layer_head_new.py --batch-size 8 --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
###
#####
#CUDA_VISIBLE_DEVICES=2 python train_adversial_UDA_with_seperated_layer_head_new.py --batch-size 16 --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=3 python train_adversial_UDA_with_seperated_layer_head_new.py --batch-size 16 --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &


#CUDA_VISIBLE_DEVICES=0 python eval.py  --device Spectralis --seg-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/tested/unet_Cirrus_to_Spectralis_nopre_ce+bce+dice+OTSU_lay_0.001_ADVENT_with_seperated_bottelnecked_layer_head_1.16v1_d_lr_1.0e-3/checkpoint/model_best.pth.tar
#CUDA_VISIBLE_DEVICES=0 python eval.py  --device Topcon --seg-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/tested/unet_Cirrus_to_Spectralis_nopre_ce+bce+dice+OTSU_lay_0.001_ADVENT_with_seperated_bottelnecked_layer_head_1.16v1_d_lr_1.0e-3/checkpoint/model_best.pth.tar
#
###
#CUDA_VISIBLE_DEVICES=1 python eval.py  --device Topcon --seg-path /home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/tested/unet_Cirrus_to_Spectralis_nopre_ce+bce+dice+OTSU_lay_0.001_ADVENT_with_seperated_bottelnecked_layer_head_1.16v1_d_lr_1.0e-3/checkpoint/model_best.pth.tar
#CUDA_VISIBLE_DEVICES=1 python eval.py  --device Cirrus --seg-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
##
####
#CUDA_VISIBLE_DEVICES=2 python eval.py  --device Cirrus --seg-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &
#CUDA_VISIBLE_DEVICES=3 python eval.py  --device Spectralis --seg-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar &

#python train_UDA_minent.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_10.31v1/checkpoint/model_best.pth.tar
#python train_UDA_minent.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_10.31v1/checkpoint/model_best.pth.tar

#python train_UDA_minent.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_10.31v1/checkpoint/model_best.pth.tar
#python train_UDA_minent.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_10.31v1/checkpoint/model_best.pth.tar


#python train_UDA_self_training.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar
#python train_UDA_self_training.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Cirrus_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v1/checkpoint/model_best.pth.tar
#
#python train_UDA_self_training.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#python train_UDA_self_training.py --device Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#
#python train_UDA_self_training.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#python train_UDA_self_training.py --device Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/unet_Topcon_nopre_NLL+dice+bce_0.001_no_dataset_norm_11.08v2/checkpoint/model_best.pth.tar
#

#python train_UDA_with_layer.py --device Topcon --source Cirrus
#python train_UDA_with_layer.py --device Topcon --source Spectralis
#python train_UDA_with_layer.py --device Cirrus --source Topcon
#python train_UDA_with_layer.py --device Cirrus --source Spectralis
#python train_UDA_with_layer.py --device Spectralis --source Topcon
#python train_UDA_with_layer.py --device Spectralis --source Cirrus
#python train_UDA_with_layer.py --device Spectralis --source Spectralis
#python train_UDA_with_intergrated_layer.py --device Cirrus --source Cirrus
#python train_UDA_with_intergrated_layer.py --device Topcon --source Topcon
#python train_UDA_with_intergrated_layer.py --device Spectralis --source Spectralis

#python train_UDA_with_layer.py --device Cirrus --source Cirrus
#python train_UDA_with_layer.py --device Topcon --source Topcon
#python train_UDA_with_layer.py --device Spectralis --source Spectralis

#python train_adversial_UDA_with_intergrated_layer.py --device Topcon --source Cirrus     --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Cirrus_to_Cirrus_nopre_ce+bce+dice+lay_0.001_with_intergrated_layer_1.10v1/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_intergrated_layer.py --device Topcon --source Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Spectralis_to_Spectralis_nopre_ce+bce+dice+lay_0.001_with_intergrated_layer_1.10v1/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_intergrated_layer.py --device Cirrus --source Topcon     --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Topcon_to_Topcon_nopre_ce+bce+dice+lay_0.001_with_intergrated_layer_1.10v1/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_intergrated_layer.py --device Cirrus --source Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Spectralis_to_Spectralis_nopre_ce+bce+dice+lay_0.001_with_intergrated_layer_1.10v1/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_intergrated_layer.py --device Spectralis --source Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Topcon_to_Topcon_nopre_ce+bce+dice+lay_0.001_with_intergrated_layer_1.10v1/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_intergrated_layer.py --device Spectralis --source Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Cirrus_to_Cirrus_nopre_ce+bce+dice+lay_0.001_with_intergrated_layer_1.10v1/checkpoint/model_best.pth.tar
#
# 9 layers
#python train_adversial_UDA_with_layer.py --device Topcon --source Cirrus     --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Cirrus_to_Cirrus_nopre_ce+bce+dice+corrected_lay_0.001_with_layer_1.11v2/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_layer.py --device Topcon --source Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Spectralis_to_Spectralis_nopre_ce+bce+dice+corrected_lay_0.001_with_layer_1.11v2/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_layer.py --device Cirrus --source Topcon     --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Topcon_to_Topcon_nopre_ce+bce+dice+corrected_lay_0.001_with_layer_1.11v2/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_layer.py --device Cirrus --source Spectralis --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Spectralis_to_Spectralis_nopre_ce+bce+dice+corrected_lay_0.001_with_layer_1.11v2/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_layer.py --device Spectralis --source Topcon --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Topcon_to_Topcon_nopre_ce+bce+dice+corrected_lay_0.001_with_layer_1.11v2/checkpoint/model_best.pth.tar
#python train_adversial_UDA_with_layer.py --device Spectralis --source Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA_new_split/result/train/Unet_with_layer_Cirrus_to_Cirrus_nopre_ce+bce+dice+corrected_lay_0.001_with_layer_1.11v2/checkpoint/model_best.pth.tar
##

#python train_UDA_minen
#python train_UDA_minent.py --device Cirrus --model-path /home/isalab102/Documents/hxx/RETOUCH_SFUDA/result/train/unet_Spectralis_nopre_NLL+dice+bce_0.001_no_dataset_norm_10.31v1/checkpoint/model_best.pth.tar
#python train_UDA_minentt.py --device Cirrus
#python train_UDA_minent.py --device Spectralis
#python train_UDA_minent.py --device Topcon

#python train_fake_SFUDA.py --device Topcon
#python train_fake_SFUDA.py --device Cirrus
#python train_fake_SFUDA.py --device Spectralis
#python eval.py --device