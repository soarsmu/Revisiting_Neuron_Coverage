import tensorflow as tf
### Path parameters
DATA_DIR = "../data/"
MODEL_DIR = "../models/"

### DATASET NAMES
MNIST = "mnist"
CIFAR = "cifar"
SVHN = "svhn"
EUROSAT = "eurosat"

DATASET_NAMES = [MNIST, CIFAR, SVHN, EUROSAT]

BENIGN = "benign"
DEEPHUNTER = "fuzzing_deephunter"
DIFFERENTIABLE = "fuzzing_differentiable"
NONDIFFERENTIABLE = "fuzzing_nondifferentiable"
SIMPLE = "simple"

### ATTACK NAMEs
BIM = "bim"
CW = "cw"
FGSM = "fgsm"
JSMA = "jsma"
PGD = "pgd"
APGD = "apgd"
DF = "deepfool"
NF = "newtonfool"
SA = "squareattack"
SHA = "shadowattack"
ST = "spatialtransformation"
WA = "wasserstein"
ATTACK_NAMES = [APGD, BIM, CW, DF, FGSM, JSMA, NF, PGD, SA, SHA, ST, WA]




### CLASSIFIER PARAMETERS
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


import tensorflow as tf
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

classifier_params = {}
for dataset_name in DATASET_NAMES :
    classifier_params[dataset_name] = {"loss_object": loss_object, "train_step": train_step}

classifier_params[MNIST]["nb_classes"] = 10
classifier_params[MNIST]["input_shape"] = (28, 28, 1)
classifier_params[MNIST]["clip_values"] = (-0.5, 0.5)

classifier_params[CIFAR]["nb_classes"] = 10
classifier_params[CIFAR]["input_shape"] = (32, 32, 3)

classifier_params[SVHN]["nb_classes"] = 10
classifier_params[SVHN]["input_shape"] = (32, 32, 3)
classifier_params[SVHN]["clip_values"] = (-0.5, 0.5)

classifier_params[EUROSAT]["nb_classes"] = 10
classifier_params[EUROSAT]["input_shape"] = (64, 64, 3)

### ATTACK PARAMETERS
attack_params = {}

attack_params[APGD] = {}
for dataset_name in DATASET_NAMES:
    attack_params[APGD][dataset_name] = {"loss_type": "cross_entropy"}

attack_params[CW] = {}
for dataset_name in DATASET_NAMES :
    attack_params[CW][dataset_name] = {}

attack_params[DF] = {}
for dataset_name in DATASET_NAMES:
    attack_params[DF][dataset_name] = {"batch_size": 256}

attack_params[NF] = {}
for dataset_name in DATASET_NAMES:
    attack_params[NF][dataset_name] = {"batch_size": 256}

attack_params[JSMA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[JSMA][dataset_name] = {}

attack_params[SA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[SA][dataset_name] = {}

attack_params[SHA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[SHA][dataset_name] = {"batch_size": 1}

attack_params[ST] = {}
for dataset_name in DATASET_NAMES:
    attack_params[ST][dataset_name] = {}

attack_params[WA] = {}
for dataset_name in DATASET_NAMES:
    attack_params[WA][dataset_name] = {}


# TODO: recheck the parameter for EUROSAT

attack_params[PGD] = {}
attack_params[PGD][MNIST] = {'eps': .3,
                      'eps_step': .03,
                      'max_iter': 20
                      }
attack_params[PGD][CIFAR] = {'eps': 16. / 255.,
                      'eps_step': 2. / 255.,
                      'max_iter': 30
                      }
attack_params[PGD][SVHN] = {'eps': 8. / 255.,
                      'eps_step': 0.01,
                      'max_iter': 30
                      }
attack_params[PGD][EUROSAT] = {'eps': 16. / 255.,
                               'eps_step': 2. / 255.,
                               'max_iter': 50
                               }

# use the same epsilon used in pgd
attack_params[BIM] = {}
attack_params[BIM][MNIST] = {'eps': .3
                                  }
attack_params[BIM][CIFAR] = {'eps': 16. / 255.
                                  }
attack_params[BIM][SVHN] = {'eps': 8. / 255.
                                 }
attack_params[BIM][EUROSAT] = {'eps': 16. / 255.
                             }


# TODO: recheck the parameter for EUROSAT

# use the same epsilon used in pgd
attack_params[FGSM] = {}
attack_params[FGSM][MNIST] = {'eps': .3
                                 }
attack_params[FGSM][CIFAR] = {'eps': 16. / 255.
                                 }
attack_params[FGSM][SVHN] = {'eps': 8. / 255.
                                }
attack_params[FGSM][EUROSAT] = {'eps': 16. / 255.
                              }

