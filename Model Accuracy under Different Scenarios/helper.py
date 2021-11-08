import time
import os
import numpy as np
from mutators import Mutators
import random
import copy
import shutil
from keras import backend as K

DATA_DIR = "../data/"
MODEL_DIR = "../models/"


# helper function
def get_layer_i_output(model, i, data):
    layer_model = K.function([model.layers[0].input], [model.layers[i].output])
    ret = layer_model([data])[0]
    num = data.shape[0]
    ret = np.reshape(ret, (num, -1))
    return ret


class Coverage:
    def __init__(self, model, x_train, y_train, x_test, y_test, x_adv):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_adv = x_adv

    # find scale factors and min num
    def scale(self, layers, batch=1024):
        data_num = self.x_adv.shape[0]
        factors = dict()
        for i in layers:
            begin, end = 0, batch
            max_num, min_num = np.NINF, np.inf
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                tmp = layer_output.max()
                max_num = tmp if tmp > max_num else max_num
                tmp = layer_output.min()
                min_num = tmp if tmp < min_num else min_num
                begin += batch
                end += batch
            factors[i] = (max_num - min_num, min_num)
        return factors

    # 1 Neuron Coverage
    def NC(self, layers, threshold=0., batch=1024):
        factors = self.scale(layers, batch=batch)
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        activate_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            buckets = np.zeros(neurons).astype('bool')
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                # scale the layer output to (0, 1)
                layer_output -= factors[i][1]
                layer_output /= factors[i][0]
                col_max = np.max(layer_output, axis=0)
                begin += batch
                end += batch
                buckets[col_max > threshold] = True
            activate_num += np.sum(buckets)
        # print('NC:\t{:.3f} activate_num:\t{} neuron_num:\t{}'.format(activate_num / neuron_num, activate_num, neuron_num))
        return activate_num / neuron_num, activate_num, neuron_num

    # 2 k-multisection neuron coverage, neuron boundary coverage and strong activation neuron coverage
    def KMNC(self, layers, k=10, batch=1024):
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        covered_num = 0
        l_covered_num = 0
        u_covered_num = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            print(neurons)
            begin, end = 0, batch
            data_num = self.x_train.shape[0]

            neuron_max = np.full(neurons, np.NINF).astype('float')
            neuron_min = np.full(neurons, np.inf).astype('float')
            while begin < data_num:
                layer_output_train = get_layer_i_output(self.model, i, self.x_train[begin:end])
                batch_neuron_max = np.max(layer_output_train, axis=0)
                batch_neuron_min = np.min(layer_output_train, axis=0)
                neuron_max = np.maximum(batch_neuron_max, neuron_max)
                neuron_min = np.minimum(batch_neuron_min, neuron_min)
                begin += batch
                end += batch
            buckets = np.zeros((neurons, k + 2)).astype('bool')
            interval = (neuron_max - neuron_min) / k
            # print(interval[8], neuron_max[8], neuron_min[8])
            begin, end = 0, batch
            data_num = self.x_adv.shape[0]
            while begin < data_num:
                layer_output_adv = get_layer_i_output(model, i, self.x_adv[begin: end])
                layer_output_adv -= neuron_min
                layer_output_adv /= (interval + 10 ** (-100))
                layer_output_adv[layer_output_adv < 0.] = -1
                layer_output_adv[layer_output_adv >= k / 1.0] = k
                layer_output_adv = layer_output_adv.astype('int')
                # index 0 for lower, 1 to k for between, k + 1 for upper
                layer_output_adv = layer_output_adv + 1
                for j in range(neurons):
                    uniq = np.unique(layer_output_adv[:, j])
                    # print(layer_output_adv[:, j])
                    buckets[j, uniq] = True
                begin += batch
                end += batch
            covered_num += np.sum(buckets[:, 1:-1])
            u_covered_num += np.sum(buckets[:, -1])
            l_covered_num += np.sum(buckets[:, 0])
        print('KMNC:\t{:.3f} covered_num:\t{}'.format(covered_num / (neuron_num * k), covered_num))
        print(
            'NBC:\t{:.3f} l_covered_num:\t{}'.format((l_covered_num + u_covered_num) / (neuron_num * 2), l_covered_num))
        print('SNAC:\t{:.3f} u_covered_num:\t{}'.format(u_covered_num / neuron_num, u_covered_num))
        return covered_num / (neuron_num * k), (l_covered_num + u_covered_num) / (
                    neuron_num * 2), u_covered_num / neuron_num, covered_num, l_covered_num, u_covered_num, neuron_num * k

    # 3 top-k neuron coverage
    def TKNC(self, layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]

        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        pattern_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            pattern_set = set()
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                # topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                for j in range(topk.shape[0]):
                    pattern_set.add(tuple(topk[j]))
                begin += batch
                end += batch
            pattern_num += len(pattern_set)
        print(
            'TKNC:\t{:.3f} pattern_num:\t{} neuron_num:\t{}'.format(pattern_num / neuron_num, pattern_num, neuron_num))
        return pattern_num / neuron_num, pattern_num, neuron_num

    # 4 top-k neuron patterns
    def TKNP(self, layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]

        def to_tuple(x):
            l = list()
            for row in x:
                l.append(tuple(row))
            return tuple(l)

        pattern_set = set()
        layer_num = len(layers)
        data_num = self.x_adv.shape[0]
        patterns = np.zeros((data_num, layer_num, k))
        layer_cnt = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                # topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                patterns[begin:end, layer_cnt, :] = topk
                begin += batch
                end += batch
            layer_cnt += 1

        for i in range(patterns.shape[0]):
            pattern_set.add(to_tuple(patterns[i]))
        pattern_num = len(pattern_set)
        print('TKNP:\t{:.3f}'.format(pattern_num))
        return pattern_num

    def all(self, layers, batch=100):
        self.NC(layers, batch=batch)
        self.KMNC(layers, batch=batch)
        self.TKNC(layers, batch=batch)
        self.TKNP(layers, batch=batch)


class AttackEvaluate:
    # model does not have softmax layer
    def __init__(self, model, ori_x, ori_y, adv_x):
        self.model = model
        # get the raw data
        self.nature_samples = ori_x
        self.labels_samples = ori_y
        # get the adversarial examples
        self.adv_samples = adv_x
        # self.adv_labels = np.load('{}{}_AdvLabels.npy'.format(self.AdvExamplesDir, self.AttackName))

        predictions = model.predict(self.adv_samples)

        def soft_max(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        tmp_soft_max = []
        for i in range(len(predictions)):
            tmp_soft_max.append(soft_max(predictions[i]))

        self.softmax_prediction = np.array(tmp_soft_max)

    # help function
    def successful(self, adv_softmax_preds, nature_true_preds):
        if np.argmax(adv_softmax_preds) != np.argmax(nature_true_preds):
            return True
        else:
            return False

    # 1 MR:Misclassification Rate
    def misclassification_rate(self):

        cnt = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
        mr = cnt / len(self.adv_samples)
        print('MR:\t\t{:.1f}%'.format(mr * 100))
        return mr

    # 2 ACAC: average confidence of adversarial class
    def avg_confidence_adv_class(self):
        cnt = 0
        conf = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                conf += np.max(self.softmax_prediction[i])

        print('ACAC:\t{:.3f}'.format(conf / cnt))
        return conf / cnt

    # 3 ACTC: average confidence of true class
    def avg_confidence_true_class(self):

        true_labels = np.argmax(self.labels_samples, axis=1)
        cnt = 0
        true_conf = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                true_conf += self.softmax_prediction[i, true_labels[i]]
        print('ACTC:\t{:.3f}'.format(true_conf / cnt))
        return true_conf / cnt

    # 4 ALP: Average L_p Distortion
    def avg_lp_distortion(self):

        ori_r = np.round(self.nature_samples * 255)
        adv_r = np.round(self.adv_samples * 255)

        NUM_PIXEL = int(np.prod(self.nature_samples.shape[1:]))

        pert = adv_r - ori_r

        dist_l0 = 0
        dist_l2 = 0
        dist_li = 0

        cnt = 0

        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                dist_l0 += (np.linalg.norm(np.reshape(pert[i], -1), ord=0) / NUM_PIXEL)
                dist_l2 += np.linalg.norm(np.reshape(self.nature_samples[i] - self.adv_samples[i], -1), ord=2)
                dist_li += np.linalg.norm(np.reshape(self.nature_samples[i] - self.adv_samples[i], -1), ord=np.inf)

        adv_l0 = dist_l0 / cnt
        adv_l2 = dist_l2 / cnt
        adv_li = dist_li / cnt

        print('**ALP:**\n\tL0:\t{:.3f}\n\tL2:\t{:.3f}\n\tLi:\t{:.3f}'.format(adv_l0, adv_l2, adv_li))
        return adv_l0, adv_l2, adv_li

    # 5 ASS: Average Structural Similarity
    def avg_SSIM(self):

        ori_r_channel = np.round(self.nature_samples * 255).astype(dtype=np.float32)
        adv_r_channel = np.round(self.adv_samples * 255).astype(dtype=np.float32)

        totalSSIM = 0
        cnt = 0

        """
        For SSIM function in skimage: http://scikit-image.org/docs/dev/api/skimage.measure.html

        multichannel : bool, optional If True, treat the last dimension of the array as channels. Similarity calculations are done 
        independently for each channel then averaged.
        """
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                totalSSIM += SSIM(X=ori_r_channel[i], Y=adv_r_channel[i], multichannel=True)

        print('ASS:\t{:.3f}'.format(totalSSIM / cnt))
        return totalSSIM / cnt

    # 6: PSD: Perturbation Sensitivity Distance
    def avg_PSD(self):

        psd = 0
        cnt = 0

        for outer in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[outer],
                               nature_true_preds=self.labels_samples[outer]):
                cnt += 1

                image = self.nature_samples[outer]
                pert = abs(self.adv_samples[outer] - self.nature_samples[outer])
                # my patch
                image = np.transpose(image, (1, 2, 0))
                pert = np.transpose(pert, (1, 2, 0))

                for idx_channel in range(image.shape[0]):
                    image_channel = image[idx_channel]
                    pert_channel = pert[idx_channel]

                    image_channel = np.pad(image_channel, 1, 'reflect')
                    pert_channel = np.pad(pert_channel, 1, 'reflect')

                    for i in range(1, image_channel.shape[0] - 1):
                        for j in range(1, image_channel.shape[1] - 1):
                            psd += pert_channel[i, j] * (1.0 - np.std(np.array(
                                [image_channel[i - 1, j - 1], image_channel[i - 1, j], image_channel[i - 1, j + 1],
                                 image_channel[i, j - 1],
                                 image_channel[i, j], image_channel[i, j + 1], image_channel[i + 1, j - 1],
                                 image_channel[i + 1, j],
                                 image_channel[i + 1, j + 1]])))
        print('PSD:\t{:.3f}'.format(psd / cnt))
        return psd / cnt

    # 7 NTE: Noise Tolerance Estimation
    def avg_noise_tolerance_estimation(self):

        nte = 0
        cnt = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                sort_preds = np.sort(self.softmax_prediction[i])
                nte += sort_preds[-1] - sort_preds[-2]

        print('NTE:\t{:.3f}'.format(nte / cnt))
        return nte / cnt

    # 8 RGB: Robustness to Gaussian Blur
    def robust_gaussian_blur(self, radius=0.5):

        total = 0
        num_gb = 0

        for i in range(len(self.adv_samples)):
            if np.argmax(self.softmax_prediction[i]) != np.argmax(self.labels_samples[i]):
                total += 1
                adv_sample = self.adv_samples[i]
                gb_sample = gaussian_blur_transform(AdvSample=adv_sample, radius=radius)
                gb_pred = self.model.predict(np.expand_dims(np.array(gb_sample), axis=0))
                if np.argmax(gb_pred) != np.argmax(self.labels_samples[i]):
                    num_gb += 1

        print('RGB:\t{:.3f}'.format(num_gb / total))
        return num_gb, total, num_gb / total

    # 9 RIC: Robustness to Image Compression
    def robust_image_compression(self, quality=50):

        total = 0
        num_ic = 0

        # prepare the save dir for the generated image(png or jpg)
        image_save = os.path.join('./tmp', 'image')
        if os.path.exists(image_save):
            shutil.rmtree(image_save)
        os.mkdir(image_save)
        # print('\nNow, all adversarial examples are saved as PNG and then compressed using *Guetzli* in the {} fold ......\n'.format(image_save))

        for i in range(len(self.adv_samples)):
            if np.argmax(self.softmax_prediction[i]) != np.argmax(self.labels_samples[i]):
                total += 1
                adv_sample = self.adv_samples[i]
                ic_sample = image_compress_transform(IndexAdv=i, AdvSample=adv_sample, dir_name=image_save,
                                                     quality=quality)
                ic_sample = np.expand_dims(ic_sample, axis=0)
                ic_pred = self.model.predict(np.array(ic_sample))
                if np.argmax(ic_pred) != np.argmax(self.labels_samples[i]):
                    num_ic += 1
        print('RIC:\t{:.3f}'.format(num_ic / total))
        return num_ic, total, num_ic / total

    def all(self):
        self.misclassification_rate()
        self.avg_confidence_adv_class()
        self.avg_confidence_true_class()
        self.avg_lp_distortion()
        self.avg_SSIM()
        self.avg_PSD()
        self.avg_noise_tolerance_estimation()
        self.robust_gaussian_blur()
        self.robust_image_compression(1)

def mutate(img, dataset):
    # ref_img is the reference image, img is the seed

    # cl means the current state of transformation
    # 0 means it can select both of Affine and Pixel transformations
    # 1 means it only select pixel transformation because an Affine transformation has been used before

    # l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
    # between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper

    # tyr_num is the maximum number of trials in Algorithm 2

    transformations = [Mutators.image_translation, Mutators.image_scale, Mutators.image_shear, Mutators.image_rotation,
                       Mutators.image_contrast, Mutators.image_brightness, Mutators.image_blur,
                       Mutators.image_pixel_change,
                       Mutators.image_noise]

    # these parameters need to be carefullly considered in the experiment
    # to consider the feedbacks
    params = []
    params.append(list(range(-3, 3)))  # image_translation
    params.append(list(map(lambda x: x * 0.1, list(range(7, 12)))))  # image_scale
    params.append(list(map(lambda x: x * 0.1, list(range(-6, 6)))))  # image_shear
    params.append(list(range(-50, 50)))  # image_rotation
    params.append(list(map(lambda x: x * 0.1, list(range(5, 13)))))  # image_contrast
    params.append(list(range(-20, 20)))  # image_brightness
    params.append(list(range(1, 10)))  # image_blur
    params.append(list(range(1, 10)))  # image_pixel_change
    params.append(list(range(1, 4)))  # image_noise

    classA = [7, 8]  # pixel value transformation
    classB = [0, 1, 2, 3, 4, 5, 6]  # Affine transformation


    x, y, z = img.shape
    random.seed(time.time())

    tid = random.sample(classA + classB, 1)[0]
    # tid = 7
    # Randomly select one transformation   Line-7 in Algorithm2
    transformation = transformations[tid]
    params = params[tid]
    # Randomly select one parameter Line 10 in Algo2
    param = random.sample(params, 1)[0]

    # Perform the transformation  Line 11 in Algo2

    # plt.imshow(img + 0.5)
    # plt.show()
    if dataset == 'cifar':
        # # for cifar dataset
        img_new = transformation(img, param)
        # img_new = np.round(img_new)
        img_new = img_new.reshape(img.shape)
    else:
        image = np.uint8(np.round((img + 0.5) * 255))
        img_new = transformation(copy.deepcopy(image), param)/ 255.0 - 0.5
        # img_new = np.round(img_new)
        img_new = img_new.reshape(img.shape)

    # Otherwise the mutation is failed. Line 20 in Algo 2
    return img_new

# the data is in range(-.5, .5)
def load_data(dataset_name):
    assert (dataset_name.upper() in ['MNIST', 'CIFAR', 'SVHN', 'FASHION_MNIST', 'OXFORD_FLOWERS102'])
    dataset_name = dataset_name.lower()
    x_train = np.load(DATA_DIR + dataset_name + '/benign/x_train.npy')
    y_train = np.load(DATA_DIR + dataset_name + '/benign/y_train.npy')
    x_test = np.load(DATA_DIR + dataset_name + '/benign/x_test.npy')
    y_test = np.load(DATA_DIR + dataset_name + '/benign/y_test.npy')
    return x_train, y_train, x_test, y_test

def check_data_path(dataset_name):
    assert os.path.exists(DATA_DIR + dataset_name + '/benign/x_train.npy')
    assert os.path.exists(DATA_DIR + dataset_name + '/benign/y_train.npy')
    assert os.path.exists(DATA_DIR + dataset_name + '/benign/x_test.npy')
    assert os.path.exists(DATA_DIR + dataset_name + '/benign/y_test.npy')

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def compare_nc(model, x_train, y_train, x_test, y_test, x_new, x_old, layer):
    l = [0, layer]
    coverage1 = Coverage(model, x_train, y_train, x_test, y_test, np.expand_dims(x_new, axis=0))
    nc1, _, _ = coverage1.NC(l, threshold=0.75, batch=1024)

    coverage2 = Coverage(model, x_train, y_train, x_test, y_test, np.expand_dims(x_old, axis=0))
    nc2, _, _ = coverage2.NC(l, threshold=0.75, batch=1024)

    if nc1 > nc2:
        return True
    else:
        return False

# the data is in range(-.5, .5)
def load_data(dataset_name):
    assert (dataset_name.upper() in ['MNIST', 'CIFAR', 'SVHN', 'EUROSAT', 'FASHION_MNIST', 'OXFORD_FLOWERS102'])
    dataset_name = dataset_name.lower()
    x_train = np.load(DATA_DIR + dataset_name + '/benign/x_train.npy')
    y_train = np.load(DATA_DIR + dataset_name + '/benign/y_train.npy')
    x_test = np.load(DATA_DIR + dataset_name + '/benign/x_test.npy')
    y_test = np.load(DATA_DIR + dataset_name + '/benign/y_test.npy')
    return x_train, y_train, x_test, y_test

def retrain(model, X_train, Y_train, X_test, Y_test, batch_size=128, epochs=50):

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # training without data augmentation
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )

    return model

# (row, col, channel)
def gaussian_blur_transform(AdvSample, radius):
    if AdvSample.shape[2] == 3:
        sample = np.round((AdvSample + 0.5) * 255)

        image = Image.fromarray(np.uint8(sample))
        gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        gb_image = np.array(gb_image).astype('float32') / 255.0 - 0.5
        # print(gb_image.shape)

        return gb_image
    else:
        sample = np.round((AdvSample + 0.5) * 255)
        sample = np.squeeze(sample, axis=2)
        image = Image.fromarray(np.uint8(sample))
        gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        gb_image = np.expand_dims(np.array(gb_image).astype('float32'), axis=-1) / 255.0 - 0.5
        # print(gb_image.shape)
        return gb_image


# use PIL Image save instead of guetzli
def image_compress_transform(IndexAdv, AdvSample, dir_name, quality=50):
    if AdvSample.shape[2] == 3:
        sample = np.round((AdvSample + .5) * 255)
        image = Image.fromarray(np.uint8(sample))
        saved_adv_image_path = os.path.join(dir_name, '{}th-adv.jpg'.format(IndexAdv))
        image.save(saved_adv_image_path, format='JPEG', quality=quality)
        IC_image = Image.open(saved_adv_image_path).convert('RGB')
        IC_image = np.array(IC_image).astype('float32') / 255.0 - .5
        return IC_image
    else:
        sample = np.round((AdvSample + .5) * 255)
        sample = np.squeeze(sample, axis=2)
        image = Image.fromarray(np.uint8(sample), mode='L')
        saved_adv_image_path = os.path.join(dir_name, '{}th-adv.jpg'.format(IndexAdv))
        image.save(saved_adv_image_path, format='JPEG', quality=quality)
        IC_image = Image.open(saved_adv_image_path).convert('L')
        IC_image = np.expand_dims(np.array(IC_image).astype('float32'), axis=-1) / 255.0 - .5
        return IC_image


# # (row, col, channel)
# def gaussian_blur_transform(AdvSample, radius):
#     if AdvSample.shape[2] == 3:
#         sample = np.round(AdvSample)
#
#         image = Image.fromarray(np.uint8(sample))
#         gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
#         gb_image = np.array(gb_image).astype('float32')
#         # print(gb_image.shape)
#
#         return gb_image
#     else:
#         sample = np.round(AdvSample)
#         sample = np.squeeze(sample, axis=2)
#         image = Image.fromarray(np.uint8(sample))
#         gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
#         gb_image = np.expand_dims(np.array(gb_image).astype('float32'), axis=-1)
#         # print(gb_image.shape)
#         return gb_image
#
#
#
# # use PIL Image save instead of guetzli
# def image_compress_transform(IndexAdv, AdvSample, dir_name, quality=50):
#     if AdvSample.shape[2] == 3:
#         sample = np.round(AdvSample)
#         image = Image.fromarray(np.uint8(sample))
#         saved_adv_image_path = os.path.join(dir_name, '{}th-adv.jpg'.format(IndexAdv))
#         image.save(saved_adv_image_path, format='JPEG', quality=quality)
#         IC_image = Image.open(saved_adv_image_path).convert('RGB')
#         IC_image = np.array(IC_image).astype('float32')
#         return IC_image
#     else:
#         sample = np.round(AdvSample)
#         sample = np.squeeze(sample, axis=2)
#         image = Image.fromarray(np.uint8(sample), mode='L')
#         saved_adv_image_path = os.path.join(dir_name, '{}th-adv.jpg'.format(IndexAdv))
#         image.save(saved_adv_image_path, format='JPEG', quality=quality)
#         IC_image = Image.open(saved_adv_image_path).convert('L')
#         IC_image = np.expand_dims(np.array(IC_image).astype('float32'), axis=-1)
#         return IC_image


