/**
 * @brief Kernelized Online Imbalanced Learning with Fixed Buddget
 * Implemented by Junjie Hu
 * Contact: jjhu@cse.cuhk.edu.hk
*/

#include "svm.h"
#include "KOIL.h"
#include <iostream>
#include <cmath>
using namespace std;


int main(int argc, char **argv)
{
    KOIL koil;

    // load data
    koil.load_data_path = "dataset/";
    koil.dataset_file = "diabetes";
    koil.idx_asso_file = koil.dataset_file+"_IdxAsso";
    koil.idx_cv_file = koil.dataset_file+"_IdxCv";

    svm_problem& prob = koil.prob;
    prob.load_problem(koil.load_data_path+koil.dataset_file);
    prob.load_cross_validation(koil.load_data_path+koil.idx_asso_file,koil.load_data_path+koil.idx_cv_file);

    // save result path and file name
    koil.save_result_path="result/";
    koil.rs_model_file = koil.save_result_path+koil.dataset_file+"_rs_model.txt";
    koil.fifo_model_file = koil.save_result_path+koil.dataset_file+"_fifo_model.txt";

    koil.rs_result_file = koil.save_result_path+koil.dataset_file+"_rs_result.txt";
    koil.fifo_result_file = koil.save_result_path+koil.dataset_file+"_fifo_result.txt";

    koil.log_file = koil.save_result_path+koil.dataset_file+"_log.txt";

    //koil.online_evaluation();


    return 0;
}
