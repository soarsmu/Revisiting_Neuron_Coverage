# declare -a attack=("autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "spatialtransformation" "squareattack")
declare -a attack=("bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "spatialtransformation" "squareattack")
# declare -a attack=("fgsm")

declare -a dataset=("mnist")
declare -a model=("lenet1" "lenet4" "lenet5")
for d in ${dataset[@]}; do
    for m in ${model[@]}; do
        for a in ${attack[@]}; do
            python coverage_curve.py \
                --dataset $d \
                --model $m \
                --attack $a
        done
    done
done

# declare -a dataset=("cifar")
# declare -a model=("vgg16" "resnet20")
# for d in ${dataset[@]}; do
#     for m in ${model[@]}; do
#         for a in ${attack[@]}; do
#             python coverage_curve.py \
#                 --dataset $d \
#                 --model $m \
#                 --attack $a
#         done
#     done
# done

# declare -a dataset=("svhn")
# declare -a model=("svhn_model" "svhn_first" "svhn_second")
# for d in ${dataset[@]}; do
#     for m in ${model[@]}; do
#         for a in ${attack[@]}; do
#             python coverage_curve.py \
#                 --dataset $d \
#                 --model $m \
#                 --attack $a
#         done
#     done
# done
