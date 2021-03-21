/*
 * This code is provided solely for the personal and private use of students
 * taking the CSC209H course at the University of Toronto. Copying for purposes
 * other than this use is expressly prohibited. All forms of distribution of
 * this code, including but not limited to public repositories on GitHub,
 * GitLab, Bitbucket, or any other online platform, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Mustafa Quraish, Bianca Schroeder, Karen Reid
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2021 Karen Reid
 */

#include "dectree.h"

/**
 * Load the binary file, filename into a Dataset and return a pointer to 
 * the Dataset. The binary file format is as follows:
 *
 *     -   4 bytes : `N`: Number of images / labels in the file
 *     -   1 byte  : Image 1 label
 *     - NUM_PIXELS bytes : Image 1 data (WIDTHxWIDTH)
 *          ...
 *     -   1 byte  : Image N label
 *     - NUM_PIXELS bytes : Image N data (WIDTHxWIDTH)
 *
 * You can set the `sx` and `sy` values for all the images to WIDTH. 
 * Use the NUM_PIXELS and WIDTH constants defined in dectree.h
 */
Dataset *load_dataset(const char *filename) {
    Dataset *dataset = malloc(sizeof(Dataset));
    FILE *f2 = fopen(filename, "rb");
    if (f2 == NULL) {
        fprintf(stderr, "Error: could not open file\n");
        exit(1); 
    }
    // allocate space
    int space;
    fread(&space, 4, 1, f2);
    dataset->num_items = space;
    dataset->images = malloc(sizeof(Image) * space);
    dataset->labels = malloc(space);
    for (int i = 0; i < space; i++){
        fread(&(dataset->labels[i]), 1, 1, f2);
        dataset->images[i].data = malloc(NUM_PIXELS);
        dataset->images[i].sy = WIDTH;
        dataset->images[i].sx = WIDTH;
        fread(dataset->images[i].data, 1, NUM_PIXELS, f2);
    }
    fclose(f2);
    return dataset;
}

/**
 * Compute and return the Gini impurity of M images at a given pixel
 * The M images to analyze are identified by the indices array. The M
 * elements of the indices array are indices into data.
 * This is the objective function that you will use to identify the best 
 * pixel on which to split the dataset when building the decision tree.
 *
 * Note that the gini_impurity implemented here can evaluate to NAN 
 * (Not A Number) and will return that value. Your implementation of the 
 * decision trees should ensure that a pixel whose gini_impurity evaluates 
 * to NAN is not used to split the data.  (see find_best_split)
 * 
 * DO NOT CHANGE THIS FUNCTION; It is already implemented for you.
 */
double gini_impurity(Dataset *data, int M, int *indices, int pixel) {
    int a_freq[10] = {0}, a_count = 0;
    int b_freq[10] = {0}, b_count = 0;

    for (int i = 0; i < M; i++) {
        int img_idx = indices[i];

        // The pixels are always either 0 or 255, but using < 128 for generality.
        if (data->images[img_idx].data[pixel] < 128) {
            a_freq[data->labels[img_idx]]++;
            a_count++;
        } else {
            b_freq[data->labels[img_idx]]++;
            b_count++;
        }
    }

    double a_gini = 0, b_gini = 0;
    for (int i = 0; i < 10; i++) {
        double a_i = ((double)a_freq[i]) / ((double)a_count);
        double b_i = ((double)b_freq[i]) / ((double)b_count);
        a_gini += a_i * (1 - a_i);
        b_gini += b_i * (1 - b_i);
    }

    // Weighted average of gini impurity of children
    return (a_gini * a_count + b_gini * b_count) / M;
}

/**
 * Given a subset of M images and the array of their corresponding indices, 
 * find and use the last two parameters (label and freq) to store the most
 * frequent label in the set and its frequency.
 *
 * - The most frequent label (between 0 and 9) will be stored in `*label`
 * - The frequency of this label within the subset will be stored in `*freq`
 * 
 * If multiple labels have the same maximal frequency, return the smallest one.
 */
void get_most_frequent(Dataset *data, int M, int *indices, int *label, int *freq) {
    // set to zero for the list
    int all_freq[10] = {0};
    for (int i = 0; i < M; i++){
        int num = indices[i];
        int label = data->labels[num];
        all_freq[label] += 1;
    }
    // find the max freq by traversing the list 
    int max_label = 0;
    int max_freq = 0;
    for (int i = 0; i < 10; i++){
        if (all_freq[i] > max_freq){
            max_freq = all_freq[i];
            max_label = i;
        }
    }
    *label = max_label;
    *freq = max_freq;
}

/**
 * Given a subset of M images as defined by their indices, find and return
 * the best pixel to split the data. The best pixel is the one which
 * has the minimum Gini impurity as computed by `gini_impurity()` and 
 * is not NAN. (See handout for more information)
 * 
 * The return value will be a number between 0-783 (inclusive), representing
 *  the pixel the M images should be split based on.
 * 
 * If multiple pixels have the same minimal Gini impurity, return the smallest.
 */
int find_best_split(Dataset *data, int M, int *indices) {
    double min_impurity = INFINITY;
    int pixel = -1;
    // traverse the list to find the label with smallest impurity
    for (int i = 0; i < 784; i++){
        double impurity = gini_impurity(data, M, indices, i);
        if (impurity < min_impurity){
            min_impurity = impurity;
            pixel = i;
        }
    }
    return pixel;
}

/**
 * Create the Decision tree. In each recursive call, we consider the subset of the
 * dataset that correspond to the new node. To represent the subset, we pass 
 * an array of indices of these images in the subset of the dataset, along with 
 * its length M. Be careful to allocate this indices array for any recursive 
 * calls made, and free it when you no longer need the array. In this function,
 * you need to:
 *
 *  Step 1:    - Compute ratio of most frequent image in indices, do not split if the
 *              ratio is greater than THRESHOLD_RATIO
 *  Step 2:    - Find the best pixel to split on using `find_best_split`
 *  Step 3:    - Split the data based on whether pixel is less than 128, allocate 
 *              arrays of indices of training images and populate them with the 
 *              subset of indices from M that correspond to which side of the split
 *              they are on
 *  Step 4:    - Allocate a new node, set the correct values and return
 *  Step 5:    - If it is a leaf node set `classification`, and both children = NULL.
 *  Step 6:    - Otherwise, set `pixel` and `left`/`right` nodes 
 *              (using build_subtree recursively). 
 */
DTNode *build_subtree(Dataset *data, int M, int *indices) {
    int label; 
    int frequency;
    int pixel;
    // get the get_most_frequent
    get_most_frequent(data, M, indices, &label, &frequency);
    double ratio = (double)frequency/M;
    int left = 0;
    int *sub_left_indices = NULL;
    int right = 0;
    int *sub_right_indices = NULL;
    int flag = -1;
    // compare the ration
    if (ratio <= THRESHOLD_RATIO) {
        pixel = find_best_split(data, M, indices);
        sub_right_indices = malloc(sizeof(int) * M);
        sub_left_indices = malloc(sizeof(int) * M);
        for (int i = 0; i < M; i++){
            int ind = indices[i];
            Image t_img = data->images[ind];
            int new_p = t_img.data[pixel];
            if (new_p < 128) {
                sub_left_indices[left] = ind;
                left++;
            } else {
                sub_right_indices[right] = ind;
                right++;
            }
        }
    }else{
        flag = label;
    }
    // recursion
    DTNode *dt_node = malloc(sizeof(DTNode));
    if (flag != -1) {
        dt_node->right = NULL;
        dt_node->left = NULL;
        dt_node->classification = flag;
        dt_node->pixel = -1;
        return dt_node;
    } 
    dt_node->classification = -1;
    dt_node->pixel = pixel;
    if (right != 0) {
        dt_node->right = build_subtree(data, right, sub_right_indices);
    } else {
        dt_node->right = NULL;
        }
    if (left != 0) {
        dt_node->left = build_subtree(data, left, sub_left_indices);
    } else {
        dt_node->left = NULL;
        }
    // free the malloc
    free(sub_right_indices);
    free(sub_left_indices); 
    
    return dt_node;
    }

/** 
 * This is the function exposed to the user. All you should do here is set
 * up the `indices` array correctly for the entire dataset and call 
 * `build_subtree()` with the correct parameters.
 */
DTNode *build_dec_tree(Dataset *data) {
    // TODO: Set up `indices` array, call `build_subtree` and return the tree.
    // HINT: Make sure you free any data that is not needed anymore
    int M = data->num_items;
    int *indices = malloc(sizeof(int) * M);
    for (int i = 0; i < M; i++){
        indices[i] = i;
    }
    // recursion
    DTNode *dt_node = build_subtree(data, M, indices);
    free(indices);
    return dt_node;
}

/**
 * Given a decision tree and an image to classify, return the predicted label.
 */
int dec_tree_classify(DTNode *root, Image *img) {
    // TODO: Return the correct label
    while (root->left != NULL || root->right != NULL){
        int check_pixel = root->pixel;
        int check_img_pixel = img->data[check_pixel];
        if (check_img_pixel < 128){
            root = root->left;
        }else{
            root = root->right;
        }
    }
    return root->classification;
}

/**
 * This function frees the Decision tree.
 */
void free_dec_tree(DTNode *node) {
    // TODO: Free the decision tree
    if (node->left != NULL || node->right != NULL){
        if(node->right != NULL){
            free_dec_tree(node->right);
        }
        if (node->left != NULL){
            free_dec_tree(node->left);
        }
    }
    free(node);
}

/**
 * Free all the allocated memory for the dataset
 */
void free_dataset(Dataset *data) {
    for (int i = 0; i < data->num_items; i ++){
        free(data->images[i].data);
    }
    free(data->labels);
    free(data->images);
    free(data);
}
// valgrand
