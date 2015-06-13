/**
 * @brief Kernelized Online Imbalanced Learning with Fixed Buddget
 * Implemented by Junjie Hu
 * Contact: jjhu@cse.cuhk.edu.hk
*/

#ifndef _SVM_H
#define _SVM_H
#include <string>
#include <vector>
using namespace std;


/**
 * @brief SVM node: <index, value> pair for each feature
 *
 */
struct svm_node
{
	int index;
	double value;
};

/**
 * @brief libsvm type dataset
 *
 */
struct svm_problem {
	// data information
	unsigned int n;		// # instances
	unsigned int pos,neg;	// # positive,negative instances
	unsigned int d;		// # features
	unsigned int elements;	// # all elemnts;

	// data value
	struct svm_node* x_space;
    struct svm_node** x;        // x[i]: the i-th sample
    double* y;                  // y[i]: the label of the i-th sample

	// cross validation partition
	int** idx_cv;
	int n_cv;
	int d_cv;
	int** idx_Asso;
	int n_Asso;
	int d_Asso;

    /**
     * @brief load samples x and label y
     *
     * @param filename file name
     */
    void load_problem(string filename);

    /**
     * @brief load the cross validation information
     *
     * @param assofile load the associated file
     * @param cvfile load the cross validation file
     */
    void load_cross_validation(string assofile, string cvfile);
};


enum {LINEAR = 0, RBF = 1, POLY = 2, SIGMOID=3};	// kernel type

/**
 * @brief contains all the SVM parameters
 *
 */
struct svm_parameter {

    svm_parameter():C(1),kernel_type(RBF),degree(2),gamma(1),eta(0.01){}

    double C;        /* penalty parameter of the objective function*/
    int kernel_type; /* kernel type: LINEAR, RBF, POLY */
    int degree;	     /* for poly */
    double gamma;	 /* for poly/rbf/sigmoid */
    //double coef0;  /* for poly/sigmoid */
    double eta;      /* learning rate for KOIL*/
};


/**
 * @brief SVM decision function
 *
 */
struct svm_model
{
    svm_model(){
        this->initialize(100);
    }
	struct svm_parameter param;	/* parameter */
	int l;			            /* total #SV */
	double b;                   // threshold for the decision
    int k_num;                  // number of k nearest neighbors
    int fpidx;                  // first index in the positive SV list for FIFO++
    int fnidx;                  // first index in the negative SV list for FIFO++
    int rsp;                    // number of scanned positive samples for RS++
    int rsn;                    // number of scanned negative samples for RS++
    double project_bound;       // project bound for the alpha
	
	// positive SV
	struct svm_node ** pos_SV;  // pos_SV: pos_SV[i][j]: the j-th feature of the i-th positive SV
	double * pos_alpha;         // alpha: weight for the positive SV
	int pos_n;                  // current number of positive SV in budget
	int max_pos_n;              // maximun number of positive SV in budget

	//negative SV
	struct svm_node ** neg_SV;  
	double * neg_alpha;
	int neg_n;
	int max_neg_n;
	
	// member function
    void initialize(int budget_size);
    void free_model();
	int load_model(string model_file_name);
	int save_model(string model_file_name);
    //double* predict(svm_problem& prob);
    double predict(svm_node* xt);
    double* predict_list(svm_node** xt, int n);
    double kernel_func(svm_node* x1, svm_node* x2);
    double f_norm();
};

/**
 * @brief SVM with Multiple Kernels decision function
 *
 */
struct svm_mkl{
    svm_mkl(){}

    svm_mkl(int budget_size, double delta, double C, vector<double> glist){
        this->initialize(budget_size,delta,C,glist);
    }

    svm_mkl(int budget_size, double delta, double C, vector<double> glist, vector<int> degreelist){
        this->initialize(budget_size,delta,C,glist,degreelist);
    }

    vector<svm_model> classifiers;
    vector<double> weight;
    vector<double> p;
    double delta;    /* smooth term */
    double lambda;   /* learning rate for each kernel of MKL */
    double eta;      /* step size for the update of kernel weight */

    //member function
    void initialize(int budget_size, double delta, double C, vector<double> glist);
    void initialize(int budget_size, double delta, double C, vector<double> glist, vector<int> degreelist);
    double predict(svm_node* xt);
    double* predict_list(svm_node** xt, int n);
    void normalize_weight();
    void smooth_propbability();
};

double dot(svm_node* a,svm_node*b);

#endif 
