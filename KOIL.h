/**
 * @brief Kernelized Online Imbalanced Learning with Fixed Buddget
 * Implemented by Junjie Hu
 * Contact: jjhu@cse.cuhk.edu.hk
*/

#ifndef _KOIL_H
#define _KOIL_H
#include <string>
#include <vector>
#include "svm.h"
using namespace std;


/**
 * @brief struct: store the AUC, Accuracy, time, error number in n runs
 *
 */
struct koil_result
{
    int runs;         //number of runs
    double* auc;      //AUC for each run
    double* accuracy; //Accuracy for each run
    double* time;     //Time for each run
    int* err_cnt;      //Misclassified samples online

    void initial_result(int n);
    void free_result();
    void save_result(string path, string method);
    void load_result(string path);
};



/**
 * @brief Kernelized Online Imbalanced Learning
 *
 */
class KOIL
{
public:
    svm_problem prob;
    svm_model rs_model;
    svm_model fifo_model;
    koil_result rs_result;
    koil_result fifo_result;

    // save and load path or file name
    string save_result_path;
    string load_data_path;
    string dataset_file;
    string idx_asso_file;
    string idx_cv_file;
    string rs_model_file;
    string fifo_model_file;
    string rs_result_file;
    string fifo_result_file;
    string log_file;

    // main functions for KOIL
    /**
     * @brief KOIL_RS++
     *
     * @param id_train the index of the training samples
     * @param cnt_train the number of the training samples
     * @param id_test the index of the testing samples
     * @param cnt_test the number of the testing samples
     * @return model the learned decision function
     * @return AUC the AUC value on the testing samples
     * @return Accuracy the Accuracy on the testing samples
     * @return time the time used for training
     * @return err_count the number of the misclassified samples online
     */
    void rs_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                 svm_model& model,double& AUC,double& Accuracy, double& time, int& err_count);

    /**
     * @brief KOIL_FIFO++
     *
     * @param id_train the index of the training samples
     * @param cnt_train the number of the training samples
     * @param id_test the index of the testing samples
     * @param cnt_test the number of the testing samples
     * @return model the learned decision function
     * @return AUC the AUC value on the testing samples
     * @return Accuracy the Accuracy on the testing samples
     * @return time the time used for training
     * @return err_count the number of the misclassified samples online
     */
    void fifo_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_model& model,double& AUC,double& Accuracy, double& time, int& err_count);

    /**
     * @brief KOIL_MKL_FIFO++
     *
     * @param id_train the index of the training samples
     * @param cnt_train the number of the training samples
     * @param id_test the index of the testing samples
     * @param cnt_test the number of the testing samples
     * @return model multiple kernels for the learned decision function
     * @return AUC the AUC value on the testing samples
     * @return Accuracy the Accuracy on the testing samples
     * @return time the time used for training
     * @return err_count the number of the misclassified samples online
     */
    void mkl_fifo_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                       svm_mkl &model,double& AUC,double& Accuracy, double& time, int& err_count);

    /**
     * @brief KOIL_MKL_RS++
     *
     * @param id_train the index of the training samples
     * @param cnt_train the number of the training samples
     * @param id_test the index of the testing samples
     * @param cnt_test the number of the testing samples
     * @return model multiple kernels for the learned decision function
     * @return AUC the AUC value on the testing samples
     * @return Accuracy the Accuracy on the testing samples
     * @return time the time used for training
     * @return err_count the number of the misclassified samples online
     */
    void mkl_rs_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                       svm_mkl &model,double& AUC,double& Accuracy, double& time, int& err_count);


    // helper functions
    /**
     * @brief KOIL_RS++: update budget
     *
     * @param xt xt the t-th sample xt
     * @param at the weight of xt
     * @param max_n the maximun number for the buffer
     * @param t the current iteration
     * @param model the current decision function f
     * @param alpha the weights of SVs, which have the same label with xt
     * @param SV the SV, which have the same label with xt
     * @param cur_n current number of SVs in the buffer
     * @param flag indicate whether xt is put in the buffer or not
     * @param ridx the replaced index for xt if xt is put in the buffer
     */
    void rs_update_budget(svm_node* xt,double at,int max_n,int t,
                          svm_model& model,double* &alpha,svm_node** &SV,int& cur_n,bool& flag, int& ridx);

    /**
     * @brief KOIL_FIFO++: update budget
     *
     * @param xt xt the t-th sample xt
     * @param at the weight of xt
     * @param max_n the maximun number for the buffer
     * @param model the current decision function f
     * @param fidx the index of the first SV in the buffer (FIFO)
     * @param alpha the weights of SVs, which have the same label with xt
     * @param SV the SV, which have the same label with xt
     * @param cur_n current number of SVs in the buffer
     * @param flag indicate whether xt is put in the buffer or not
     * @param ridx the replaced index for xt if xt is put in the buffer
     */
    void fifo_update_budget(svm_node* xt,double at,int max_n,
                            svm_model& model,int& fidx,double* &alpha,svm_node** &SV,int& cur_n,bool& flag, int& ridx);

    /**
     * @brief KOIL: update threshold of decision function
     *
     * @param model the current decision function
     */
    void update_b(svm_model& model);

    /**
     * @brief update the weight for SV
     *
     * @param xt the t-th sample xt
     * @param yt the label of xt
     * @param model the current decision function f
     * @return at return the weight of xt
     * @param losstype indicate l1 or l2 loss, default = "l1"
     */
    void update_kernel(svm_node* xt,double yt, svm_model& model, double& at);

    /**
     * @brief update the weight for SV
     *
     * @param xt the t-th sample xt
     * @param yt the label of xt
     * @param model the current decision function f
     * @return at return the weight of xt
     * @param losstype indicate l1 or l2 loss, default = "l1"
     */
    void update_kernel_l2(svm_node* xt,double yt, svm_model& model, double& at);

    /**
     * @brief the calculate the AUC and Accuracy between f and y
     *
     * @param f 1xn vector, the predicted label by the model
     * @param y 1xn vector, the true label
     * @param n the number of the label
     * @return AUC AUC value
     * @return Accuracy Accuracy for the correct prediction
     */
    void evaluate_AUC(double* f, double* y, int n,
                      double& AUC, double& Accuracy);


};

#endif
