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


    # mnist for defended model
    python attack.py --dataset mnist --model adv_lenet1_fgsm --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet1_pgd --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet1_apgd --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet1_deephunter --attack $attack --batch_size 256

    python attack.py --dataset mnist --model adv_lenet4_fgsm --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet4_pgd --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet4_apgd --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet4_deephunter --attack $attack --batch_size 256

    python attack.py --dataset mnist --model adv_lenet5_fgsm --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet5_pgd --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet5_apgd --attack $attack --batch_size 256
    python attack.py --dataset mnist --model adv_lenet5_deephunter --attack $attack --batch_size 256

    # cifar for defended model
    python attack.py --dataset cifar --model adv_vgg16_fgsm --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_vgg16_pgd --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_vgg16_apgd --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_vgg16_deephunter --attack $attack --batch_size 256

    python attack.py --dataset cifar --model adv_resnet20_fgsm --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_resnet20_pgd --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_resnet20_apgd --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_resnet20_deephunter --attack $attack --batch_size 256

    python attack.py --dataset cifar --model adv_resnet56_fgsm --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_resnet56_pgd --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_resnet56_apgd --attack $attack --batch_size 256
    python attack.py --dataset cifar --model adv_resnet56_deephunter --attack $attack --batch_size 256


    # svhn for defended model
    python attack.py --dataset svhn --model adv_svhn_model_fgsm --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_model_pgd --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_model_apgd --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_model_deephunter --attack $attack --batch_size 256

    python attack.py --dataset svhn --model adv_svhn_first_fgsm --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_first_pgd --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_first_apgd --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_first_deephunter --attack $attack --batch_size 256

    python attack.py --dataset svhn --model adv_svhn_second_fgsm --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_second_pgd --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_second_apgd --attack $attack --batch_size 256
    python attack.py --dataset svhn --model adv_svhn_second_deephunter --attack $attack --batch_size 256

    # eurosat for defended model
    python attack.py --dataset eurosat --model adv_resnet20_fgsm --attack $attack --batch_size 256
    python attack.py --dataset eurosat --model adv_resnet20_pgd --attack $attack --batch_size 256
    python attack.py --dataset eurosat --model adv_resnet20_apgd --attack $attack --batch_size 256
    python attack.py --dataset eurosat --model adv_resnet20_deephunter --attack $attack --batch_size 256

    python attack.py --dataset eurosat --model adv_resnet56_fgsm --attack $attack --batch_size 256
    python attack.py --dataset eurosat --model adv_resnet56_pgd --attack $attack --batch_size 256
    python attack.py --dataset eurosat --model adv_resnet56_apgd --attack $attack --batch_size 256
    python attack.py --dataset eurosat --model adv_resnet56_deephunter --attack $attack --batch_size 256

done    
