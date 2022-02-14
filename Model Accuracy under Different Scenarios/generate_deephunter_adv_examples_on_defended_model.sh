# mnist for defended model
python deephunter_attack.py --dataset mnist --model adv_lenet1_fgsm 
python deephunter_attack.py --dataset mnist --model adv_lenet1_pgd 
python deephunter_attack.py --dataset mnist --model adv_lenet1_apgd 
python deephunter_attack.py --dataset mnist --model adv_lenet1_deephunter 

python deephunter_attack.py --dataset mnist --model adv_lenet4_fgsm 
python deephunter_attack.py --dataset mnist --model adv_lenet4_pgd 
python deephunter_attack.py --dataset mnist --model adv_lenet4_apgd 
python deephunter_attack.py --dataset mnist --model adv_lenet4_deephunter 

python deephunter_attack.py --dataset mnist --model adv_lenet5_fgsm 
python deephunter_attack.py --dataset mnist --model adv_lenet5_pgd 
python deephunter_attack.py --dataset mnist --model adv_lenet5_apgd 
python deephunter_attack.py --dataset mnist --model adv_lenet5_deephunter 

# cifar for defended model
python deephunter_attack.py --dataset cifar --model adv_vgg16_fgsm 
python deephunter_attack.py --dataset cifar --model adv_vgg16_pgd 
python deephunter_attack.py --dataset cifar --model adv_vgg16_apgd 
python deephunter_attack.py --dataset cifar --model adv_vgg16_deephunter 

python deephunter_attack.py --dataset cifar --model adv_resnet20_fgsm 
python deephunter_attack.py --dataset cifar --model adv_resnet20_pgd 
python deephunter_attack.py --dataset cifar --model adv_resnet20_apgd 
python deephunter_attack.py --dataset cifar --model adv_resnet20_deephunter 

python deephunter_attack.py --dataset cifar --model adv_resnet56_fgsm 
python deephunter_attack.py --dataset cifar --model adv_resnet56_pgd 
python deephunter_attack.py --dataset cifar --model adv_resnet56_apgd 
python deephunter_attack.py --dataset cifar --model adv_resnet56_deephunter 


# svhn for defended model
python deephunter_attack.py --dataset svhn --model adv_svhn_model_fgsm 
python deephunter_attack.py --dataset svhn --model adv_svhn_model_pgd 
python deephunter_attack.py --dataset svhn --model adv_svhn_model_apgd 
python deephunter_attack.py --dataset svhn --model adv_svhn_model_deephunter 

python deephunter_attack.py --dataset svhn --model adv_svhn_first_fgsm 
python deephunter_attack.py --dataset svhn --model adv_svhn_first_pgd 
python deephunter_attack.py --dataset svhn --model adv_svhn_first_apgd 
python deephunter_attack.py --dataset svhn --model adv_svhn_first_deephunter 

python deephunter_attack.py --dataset svhn --model adv_svhn_second_fgsm 
python deephunter_attack.py --dataset svhn --model adv_svhn_second_pgd 
python deephunter_attack.py --dataset svhn --model adv_svhn_second_apgd 
python deephunter_attack.py --dataset svhn --model adv_svhn_second_deephunter 

# eurosat for defended model
python deephunter_attack.py --dataset eurosat --model adv_resnet20_fgsm 
python deephunter_attack.py --dataset eurosat --model adv_resnet20_pgd 
python deephunter_attack.py --dataset eurosat --model adv_resnet20_apgd 
python deephunter_attack.py --dataset eurosat --model adv_resnet20_deephunter 

python deephunter_attack.py --dataset eurosat --model adv_resnet56_fgsm 
python deephunter_attack.py --dataset eurosat --model adv_resnet56_pgd 
python deephunter_attack.py --dataset eurosat --model adv_resnet56_apgd 
python deephunter_attack.py --dataset eurosat --model adv_resnet56_deephunter 
