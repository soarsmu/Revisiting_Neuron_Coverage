# mnist dataset
python deephunter_attack.py --dataset mnist --model lenet1
python deephunter_attack.py --dataset mnist --model lenet4
python deephunter_attack.py --dataset mnist --model lenet5

# cifar dataset
python deephunter_attack.py --dataset cifar --model vgg16
python deephunter_attack.py --dataset cifar --model resnet20
python deephunter_attack.py --dataset cifar --model resnet56

# svhn dataset
python deephunter_attack.py --dataset svhn --model svhn_model
python deephunter_attack.py --dataset svhn --model svhn_first
python deephunter_attack.py --dataset svhn --model svhn_second

# eurosat dataset
python deephunter_attack.py --dataset eurosat --model resnet20
python deephunter_attack.py --dataset eurosat --model resnet56