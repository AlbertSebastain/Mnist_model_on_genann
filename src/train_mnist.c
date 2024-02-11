#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include "genann.h"

#define TEST_DATASET_NAME "../dataset/t10k-images.idx3-ubyte"
#define TEST_DATASET_LABEL_NAME "../dataset/t10k-labels.idx1-ubyte"
#define TRAIN_DATASET_NAME "../dataset/train-images.idx3-ubyte"
#define TRAIN_DATASET_LABEL_NAME "../dataset/train-labels.idx1-ubyte"
#define BEST_NET_PATH "../nets/best_net.bin"

struct dataset_info {
    uint32_t magic_number;
    uint32_t data_size;
    uint32_t row;
    uint32_t col;
} __attribute__((packed));

struct train_parameters {
    int hidden_size;
    int hidden_layers;
    int epochs;
    int learning_rate;
};

#define B2S(num) num = (((num & 0xff000000) >> 24) | \
                        ((num & 0x00ff0000) >> 8) | \
                        ((num & 0x0000ff00) << 8) | \
                        ((num & 0x000000ff) << 24))

bool load_data_eval(char *test_data_filename, char *test_label_filename, genann *net);
double eval_net(genann *net, double **images, uint8_t *labels, struct dataset_info *info);

bool load_dataset_info(FILE *p, struct dataset_info *info)
{
    if (p == NULL || info == NULL) {
        return false;
    }

    size_t len = 0;
    len = fread(info, sizeof(struct dataset_info), 1, p);
    if (len == 0) {
        return false;
    }
    B2S(info->magic_number);
    B2S(info->data_size);
    B2S(info->row);
    B2S(info->col);
    printf("data set information: data size: %d row: %d col: %d\n", info->data_size, info->row, info->col);
    return true;
}

void free_images(double **images, uint32_t len) {
    if (images == NULL) {
        return;
    }
    for (uint32_t i = 0; i < len; i++) {
        if (images[i] == NULL) {
            continue;
        }
        free(images[i]);
    }
    free(images);
    return;
}

double **load_images(FILE *p, struct dataset_info *info)
{
    if (p == NULL || info == NULL) {
        return NULL;
    }
    size_t len = 0;
    double **images = malloc(info->data_size * sizeof(double *));
    if (images == NULL) {
        printf("malloc failed\n");
        return NULL;
    }
    memset(images, 0x0, sizeof(info->data_size * sizeof(double *)));
    uint8_t *image = malloc(info->row * info->col * sizeof(uint8_t));
    if (image == NULL) {
        free(images);
        return NULL;
    }
    double *desired_image = NULL;
    for (uint32_t i = 0; i < info->data_size; i++) {
        desired_image = malloc(info->row * info->col * sizeof(double));
        if (desired_image == NULL) {
            free(image);
            free(images);
            images = NULL;
            printf("malloc failed\n");
            return NULL;
        }
        len = fread(image, sizeof(uint8_t), (info->row * info->col * sizeof(uint8_t)), p);
        if (len == 0) {
            free(desired_image);
            free_images(images, info->data_size);
            images = NULL;
            printf("read %d image failed\n", i);
            return NULL;
        }
        for (uint32_t j = 0; j < info->row * info->col; j++) {
            desired_image[j] = (double)image[j] / 255;
        }
        images[i] = desired_image;
    }
    return images;
}

uint8_t *load_labels(FILE *p, struct dataset_info *info) {
    if (p == NULL || info == NULL) {
        return NULL;
    }

    uint8_t *labels = malloc(info->data_size * sizeof(uint8_t));
    if (labels == NULL) {
        return NULL;
    }

    uint32_t data[2];
    size_t len = 0;
    len = fread(data, sizeof(uint32_t), 2, p);
    if (len == 0) {
        free(labels);
        labels = NULL;
        return NULL;
    }
    B2S(data[1]);
    printf("labels size: %d\n", data[1]);
    if (data[1] != info->data_size) {
        printf("the expected size: %d\n", info->data_size);
        free(labels);
        labels = NULL;
        return NULL;
    }
    len = fread(labels, sizeof(uint8_t), info->data_size, p);
    if (len == 0) {
        printf("load labels failed\n");
        free(labels);
        labels = NULL;
        return NULL;
    }
    return labels;
}

void Knuth_Durstenfeld_Shuffle(uint32_t *arr, int32_t len)
{
    for (int i = len; i >= 1; --i) {
        srand(time(NULL));
        uint8_t temp;
        int idx = rand()%(i+1);
        temp = arr[idx];
        arr[idx] = arr[i];
        arr[i] = temp;
    }
}

double *convert_to_onehot(uint8_t *labels, uint32_t size, uint32_t max_val) {
    double *onehot = (double*)malloc(size * max_val * sizeof(double));
    if (onehot == NULL) {
        printf("Memory allocation failed\n");
        return NULL;
    }

    for (uint32_t i = 0; i < size; i++) {
        for (uint32_t j = 0; j < max_val; j++) {
            if (j == labels[i]) {
                onehot[i * max_val + j] = 1.0;
            } else {
                onehot[i * max_val + j] = 0.0;
            }
        }
    }

    return onehot;
}

genann *train(double **images, double *onehot_labs, double **validate_images, uint8_t *validate_labels, struct dataset_info *validate_info,
        struct dataset_info *info, struct train_parameters *params)
{
    double best_accuracy = 0.0;
    double accuracy;
    genann *best_net = NULL;
    if (images == NULL || onehot_labs == NULL || params == NULL) {
        return NULL;
    }
    uint32_t *idx = malloc(sizeof(uint32_t) * info->data_size);
    if (idx == NULL) {
        return NULL;
    }
    for (int i = 0; i < info->data_size; i++) {
        idx[i] = i;
    }
    genann *net = genann_init(info->row * info->col, params->hidden_layers, params->hidden_size, 10);
    for (uint32_t epoch = 0; epoch < params->epochs; epoch++) {
        Knuth_Durstenfeld_Shuffle(idx, info->data_size);
        for (uint32_t iter = 0; iter < info->data_size; iter++) {
            genann_train(net, images[idx[iter]], onehot_labs + 10 * idx[iter], params->learning_rate);
            if (iter % 30000 == 0) {
                accuracy = eval_net(net, validate_images, validate_labels, validate_info);
                if (best_accuracy < accuracy) {
                    best_accuracy = accuracy;
                    if (best_net != NULL) {
                        genann_free(best_net);
                    }
                    best_net = genann_copy(net);
                }
                printf("Training epoch %d iteration %d accuracy on validation set: %f, best accurary %f\n", epoch, iter, accuracy, best_accuracy);
            }
        }
    }
    return best_net;
}

bool load_data_from_file(char *data_filename, char *label_filename, struct dataset_info *info, uint8_t **labels, double ***images)
{
    bool ret = false;
    FILE *image_f = fopen(data_filename, "rb");
    if (image_f == NULL) {
        return ret;
    }
    FILE *label_f = fopen(label_filename, "rb");
    if (label_f == NULL) {
        goto err_0;
    }
    ret = load_dataset_info(image_f, info);
    if (!ret) {
        goto err_1;
    }
    *images = load_images(image_f, info);
    if (*images == NULL) {
        ret = false;
        goto err_1;
    }
    *labels = load_labels(label_f, info);
    if (*labels == NULL) {
        ret = false;
        goto err_1;
    }
err_1:
    fclose(label_f);
err_0:
    fclose(image_f);
    return ret;
}

void split_validate_test_set(double **test_images, uint8_t *test_labels, double ***validate_images, uint8_t **validate_labels,
    struct dataset_info *test_info, struct dataset_info *validate_info, uint32_t ratio)
{
    uint32_t validate_size = test_info->data_size / ratio;
    *validate_images = test_images + test_info->data_size - validate_size;
    *validate_labels = test_labels + test_info->data_size - validate_size;
    memcpy(validate_info, test_info, sizeof(struct dataset_info));
    test_info->data_size = test_info->data_size - validate_size;
    validate_info->data_size = validate_size;
}

genann *load_data_train_net(char *data_filename, char *label_filename,
        double **validate_images, uint8_t *validate_labels, struct dataset_info *validate_info, struct train_parameters *params)
{
    bool ret = false;
    uint8_t *labels;
    double **images;
    double *onehot_labels;
    struct dataset_info info;
    genann *net = NULL;
    ret = load_data_from_file(data_filename, label_filename, &info, &labels, &images);
    if (!ret) {
        printf("load failed\n");
        return NULL;
    }
    onehot_labels = convert_to_onehot(labels, info.data_size, 10);
    if (onehot_labels == NULL) {
        printf("convert failed\n");
        return NULL;
    }
    net = train(images, onehot_labels, validate_images, validate_labels, validate_info, &info, params);
    return net;
}

int argmax(double *outputs, int len)
{
    double max_val = outputs[0];
    int idx = 0;
    for (int i = 1; i < len; i++) {
        //printf("outputs[%d] = %f", i, outputs[i]);
        if (max_val < outputs[i]) {
            max_val = outputs[i];
            idx = i;
        }
    }
    return idx;
}

double eval_net(genann *net, double **images, uint8_t *labels, struct dataset_info *info)
{
    int hit = 0;

    for (uint32_t item = 0; item < info->data_size; ++item) {
        double *test_res = (double *)genann_run(net, images[item]);
        int class = argmax(test_res, 10);
        if (class == (int)labels[item]) {
            hit++;
        }
    }
    double accuracy = (double) hit / (double) info->data_size;
    return accuracy;
}

bool load_test_validate_set(char *test_data_filename, char *test_label_filename, double ***test_images, uint8_t **test_labels,
    double ***validate_images, uint8_t **validate_labels, struct dataset_info *test_info, struct dataset_info *validate_info)
{
    bool ret = false;
    ret = load_data_from_file(test_data_filename, test_label_filename, test_info, test_labels, test_images);
    if (!ret) {
        printf("load failed\n");
        return ret;
    }
    split_validate_test_set(*test_images, *test_labels, validate_images, validate_labels, test_info, validate_info, 5);
    return true;
}


int main()
{
    struct train_parameters params = {
        .epochs = 20,
        .hidden_layers = 2,
        .hidden_size = 15,
        .learning_rate = 2,
    };
    genann *net;

    double **test_images;
    double **validate_images;
    uint8_t *test_labels;
    uint8_t *validate_labels;
    struct dataset_info test_info;
    struct dataset_info validate_info;

    load_test_validate_set(TEST_DATASET_NAME, TEST_DATASET_LABEL_NAME, &test_images, &test_labels, &validate_images, &validate_labels, &test_info, &validate_info);
    net = load_data_train_net(TRAIN_DATASET_NAME, TRAIN_DATASET_LABEL_NAME, validate_images, validate_labels, &validate_info, &params);
    double accuracy = eval_net(net, test_images, test_labels, &test_info);
    printf("Accuracy on test set %f\n", accuracy);
    FILE *save_net_p = fopen(BEST_NET_PATH, "wb");
    if (save_net_p == NULL) {
        printf("Open for saving net failed\n");
        return 0;
    }
    genann_write(net, save_net_p);
    return 0;
}