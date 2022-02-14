declare -a attacks=("fgsm" "deepfool" "pgd" "apgd" "bim" "cw" )

for attack in ${attacks[@]}; do

    # mnist dataset
    python attack.py --dataset mnist --model lenet1 --attack $attack --batch_size 256
    python attack.py --dataset mnist --model lenet4 --attack $attack --batch_size 256
    python attack.py --dataset mnist --model lenet5 --attack $attack --batch_size 256

    # cifar dataset
    python attack.py --dataset cifar --model vgg16 --attack $attack --batch_size 256
    python attack.py --dataset cifar --model resnet20 --attack $attack --batch_size 256
    python attack.py --dataset cifar --model resnet56 --attack $attack --batch_size 256

    # svhn dataset
    python attack.py --dataset svhn --model svhn_model --attack $attack --batch_size 256
    python attack.py --dataset svhn --model svhn_first --attack $attack --batch_size 256
    python attack.py --dataset svhn --model svhn_second --attack $attack --batch_size 256

    # eurosat dataset
    python attack.py --dataset eurosat --model resnet20 --attack $attack --batch_size 256
    python attack.py --dataset eurosat --model resnet56 --attack $attack --batch_size 256

done    
