/**
 * @brief Kernelized Online Imbalanced Learning with Fixed Buddget
 * Implemented by Junjie Hu
 * Contact: jjhu@cse.cuhk.edu.hk
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <cstdarg>
#include <limits>
#include <locale>
#include "svm.h"
#include "utility.h"

using namespace std;

static const char *kernel_type_table[]=
{
	"linear","rbf","polynomial","sigmoid",NULL
};

/****************************************
 * svm_model: save and load svm_model
 * **************************************/
// initial svm model
void svm_model::initialize(int budget_size)
{
    pos_n = neg_n = l = 0;
    max_pos_n = max_neg_n = budget_size;
    k_num = 0.05*budget_size;
    pos_alpha = Malloc(double,budget_size);
    neg_alpha = Malloc(double,budget_size);
    memset(pos_alpha,0,budget_size);
    memset(neg_alpha,0,budget_size);
    pos_SV = Malloc(svm_node*,budget_size);
    neg_SV = Malloc(svm_node*,budget_size);
    b = 0;
    fpidx = 0;
    fnidx = 0;
}

// free the svm model
void svm_model::free_model()
{
    free(pos_alpha);
    free(neg_alpha);
    free(pos_SV);
    free(neg_SV);
}

// save svm model
int svm_model::save_model(string model_file_name)
{
	FILE *fp = fopen(model_file_name.c_str(),"w");
	if(fp==NULL) return -1;
	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);
	fprintf(fp,"C %.8g\n",param.C);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %g\n", param.gamma);

	fprintf(fp, "total_sv %d\n",l);           // number of SVs
	fprintf(fp, "threshold %.8g\n",b);          // decision threshold
	
	fprintf(fp, "SV\n");
	// positive SV
	fprintf(fp,"pos_n %d\n",pos_n);
	fprintf(fp,"max_pos_n %d\n",max_pos_n);
	fprintf(fp,"pos_SV\n");
	for(int i=0;i<pos_n;i++)
	{
		fprintf(fp,"%.16g ",pos_alpha[i]);
		const svm_node *p = pos_SV[i];

		while(p->index !=-1)
		{
			fprintf(fp,"%d:%.8g ",p->index,p->value);
			p++;
		}
		fprintf(fp, "\n");
	}

	// negative SV
	fprintf(fp,"neg_n %d\n",neg_n);
	fprintf(fp,"max_neg_n %d\n",max_neg_n);
	fprintf(fp,"neg_SV\n");
	for(int i=0;i<neg_n;i++)
	{
		fprintf(fp,"%.16g ",neg_alpha[i]);
		const svm_node *p = neg_SV[i];

		while(p->index !=-1)
		{
			fprintf(fp,"%d:%.8g ",p->index,p->value);
			p++;
		}
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

// load svm model (unfinished)
int svm_model::load_model(string model_file_name)
{
//	FILE *fp = fopen(model_file_name.c_str(),"rb");
//	if(fp==NULL) return -1;
//	char *old_locale = strdup(setlocale(LC_ALL, NULL));
//	setlocale(LC_ALL, "C");

//	char cmd[101];
//	// read parameters
//	while(1)
//	{
//		fscanf(fp,"%100s",cmd);

//		if(strcmp(cmd,"kernel_type")==0)
//		{
//			fscanf(fp,"%100s",cmd);
//			int i;
//			for(i=0;kernel_type_table[i];i++)
//			{
//				if(strcmp(kernel_type_table[i],cmd)==0)
//				{
//					param.kernel_type=i;
//					break;
//				}
//			}
//			if(kernel_type_table[i]==NULL)
//			{
//				fprintf(stderr,"unknown kernel function.\n");
//				setlocale(LC_ALL, old_locale);
//				return -1;
//			}
//		}
//		else if(strcmp(cmd,"C")==0)
//			fscanf(fp,"%d",&param.C);
//		else if(strcmp(cmd,"degree")==0)
//			fscanf(fp,"%d",&param.degree);
//		else if(strcmp(cmd,"gamma")==0)
//			fscanf(fp,"%d",&param.gamma);
//		else if(strcmp(cmd,"total_sv")==0)
//			fscanf(fp,"%d",&l);
//		else if(strcmp(cmd,"SV")==0)
//		{
//			while(1)
//			{
//				int c = getc(fp);
//				if(c==EOF || c=='\n') break;
//			}
//			break;
//		}
//		else{
//			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
//			return -1;
//		}
//	}

//	// read sv_coef and SV

//	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);
//	fprintf(fp,"C %.8g\n",param.C);

//	if(param.kernel_type == POLY)
//		fprintf(fp,"degree %d\n", param.degree);

//	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
//		fprintf(fp,"gamma %g\n", param.gamma);

//	fprintf(fp, "total_sv %d\n",l);           // number of SVs
//	fprintf(fp, "threshold %.8g\n",b);          // decision threshold
	
//	fprintf(fp, "SV\n");
//	// positive SV
//	fprintf(fp,"pos_n %d\n",pos_n);
//	fprintf(fp,"max_pos_n %d\n",max_pos_n);
//	fprintf(fp,"pos_SV\n");
//	for(int i=0;i<pos_n;i++)
//	{
//		fprintf(fp,"%.16g ",pos_alpha[i]);
//		const svm_node *p = pos_SV[i];

//		while(p->index !=-1)
//		{
//			fprintf(fp,"%d:%.8g ",p->index,p->value);
//			p++;
//		}
//		fprintf(fp, "\n");
//	}

//	// negative SV
//	fprintf(fp,"neg_n %d\n",neg_n);
//	fprintf(fp,"max_neg_n %d\n",max_neg_n);
//	fprintf(fp,"neg_SV\n");
//	for(int i=0;i<neg_n;i++)
//	{
//		fprintf(fp,"%.16g ",neg_alpha[i]);
//		const svm_node *p = neg_SV[i];

//		while(p->index !=-1)
//		{
//			fprintf(fp,"%d:%.8g ",p->index,p->value);
//			p++;
//		}
//		fprintf(fp, "\n");
//	}

//	setlocale(LC_ALL, old_locale);
//	free(old_locale);

//	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
//	else return 0;
    return 0;
}

// predict the label for one node
double svm_model::predict(svm_node* xt)
{
    double probability=0;
//    if(this->pos_n>=this->max_pos_n)
//        cerr<<"the current no. of positive sv ("<<pos_n<<") excess the maximun no. ("<<max_pos_n<<")."<<endl;
//    if(this->neg_n>=this->max_neg_n)
//        cerr<<"the current no. of negative sv ("<<neg_n<<") excess the maximun no. ("<<max_neg_n<<")."<<endl;

    for(int i=0;i<this->pos_n;i++){
        probability += this->pos_alpha[i]*kernel_func(this->pos_SV[i],xt);
    }
    for(int i=0;i<this->neg_n;i++){
        probability += this->neg_alpha[i]*kernel_func(this->neg_SV[i],xt);
    }
    probability -= this->b;
    return probability;
}

// predict the label for one node
double* svm_model::predict_list(svm_node** xt, int n)
{
    double* plist = Malloc(double,n);
    for(int i=0;i<n;i++){
        plist[i] = predict(xt[i]);
    }
    return plist;
}

double svm_model::kernel_func(svm_node* a, svm_node* b)
{
    double ans=0;
    double temp=0;
    switch(param.kernel_type){
    case LINEAR:
    {
        return dot(a,b);
    }
    case RBF:
    {
        while(a->index !=-1 || b->index !=-1){
            if(a->index == b->index){
                temp = a->value - b->value;
                ans += (temp*temp);
//                cout<<"a index "<<a->index<<":value "<<a->value<<endl;
//                cout<<"b index "<<b->index<<":value "<<b->value<<endl;
                ++a;
                ++b;
            }else if(a->index==-1 && b->index!=-1){
                ans += (b->value*b->value);
                ++b;
            }else if(b->index==-1 && a->index!=-1){
                ans += (a->value*a->value);
                ++a;
            }else if(a->index > b->index){
                ans += (b->value * b->value);
//                cout<<"b index "<<b->index<<":value "<<b->value<<endl;
                ++b;
            }else if(b->index > a->index){
                ans += (a->value * a->value);
//                cout<<"a index "<<a->index<<":value "<<a->value<<endl;
                ++a; 
            }
        }
//        cout<<"norm = "<<ans<<endl;
        ans = std::exp(-1*ans*param.gamma);
//        cout<<"kernel value = "<<ans<<endl;
        return ans;
    }
    case POLY:
    {

    }
    }
}


/****************************************
 * svm_problem:
 * load svm_problem and cross validation file
 * **************************************/
// load data
void svm_problem::load_problem(string filename)
{
	unsigned int max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename.c_str(),"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		cout<<"can't open input file" << filename<<endl;
		exit(1);
	}

	this->n = 0;
	this->elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++this->elements;
		}
		++this->elements;
		++this->n;
	}
	rewind(fp);

	this->pos = this->neg = 0;
	this->x_space = Malloc(svm_node,this->elements + this->n);
	this->x = Malloc(svm_node*,this->n);
	this->y = Malloc(double,this->n);
	max_index = j = 0;

	// read data 
	j = 0;
	for(i = 0; i < this->n; ++i)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		this->x[i] = &this->x_space[j];  // x and x_space point to the same space
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			cout << "lib file label format error1" << endl;

		this->y[i] = strtod(label,&endptr);
		if (this->y[i] != 1) {
			this->y[i] = -1.0;
			++this->neg;
		} else {
			++this->pos;
		}

		if(endptr == label || *endptr != '\0')
			cout << "lib file label format error2" << endl;

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			this->x_space[j].index = (unsigned int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || this->x_space[j].index <= inst_max_index)
				cout << "lib file format error1" << endl;;

			errno = 0;
			this->x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				cout << "lib file format error2" << endl;

			if (fabs(this->x_space[j].value) > 1e-16) {
				inst_max_index = this->x_space[j].index;
				++j;
			}
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		this->x_space[j++].index = -1;   // index = -1 means the end of this SV
	}
	fclose(fp);

	this->d = max_index;
}

// load cross_validation file
void svm_problem::load_cross_validation(string assofile, string cvfile)
{
    read_matrix(assofile,this->idx_Asso, this->n_Asso, this->d_Asso);
    read_matrix(cvfile,this->idx_cv, this->n_cv, this->d_cv);
}

double dot(svm_node* a,svm_node*b)
{
    double ans = 0.0;
    while(a->index !=-1 && b->index !=-1){
        if(a->index == b->index){
            ans += a->value*b->value;
            ++a;
            ++b;
        }else if(a->index > b->index)
            ++b;
        else
            ++a;
    }
    return ans;
}


