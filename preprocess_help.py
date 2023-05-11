import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

from tubular.base import DataFrameMethodTransformer
from tubular.capping import CappingTransformer
from tubular.imputers import NearestMeanResponseImputer, MeanImputer, NullIndicator, ArbitraryImputer
#from tubular.mapping import MappingTransformer
from tubular.nominal import GroupRareLevelsTransformer, OneHotEncodingTransformer, MeanResponseTransformer
from tubular.nominal import NominalToIntegerTransformer
from tubular.numeric import CutTransformer, ScalingTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from scipy import stats
from scipy.stats import kurtosis, skew
from IPython.display import clear_output
from time import sleep


class DataPreProcess:
    def __init__(self, df):
        '''
        Initialize DataPreProcess class. bring in data frame and initialize lists
        '''
        self.df = df
        self.skip_cols = []
        self.id_cols = []
        self.float_cols = []
        self.cat_cols = []
        self.ordinal_cols = []
        self.other_cols = []
        self.binary_cols = []
        self.response_dict = {}
        self.response_cols = []
        self.skew_transformer = {}
        self.cat_decision = 'one-hot'
        self.model_flag = 'no model yet'

        self.feature_options = ['skip', 'id', 'response', 'float', 'category', 'binary', 'ordinal', 'other']
        
    #@staticmethod
    def fun_progress_in_loop(i_iter, len_iter):
        percent_color_text = '\x1b[5;37;42m'
        progress_bar = (i_iter+1)/len_iter
        num_green_spaces = " " * int(progress_bar * 50)
        num_remaining_space = " " * (50 - int(progress_bar * 50))
        percent_left = str( int(progress_bar*100) )
        print(percent_left + "%\t[" + percent_color_text + num_green_spaces + "\x1b[0m" + 
              num_remaining_space + "] {}/{} columns".format((i_iter+1), len_iter))
        
        
    def process_columns(self, single_col='no column specified'):
        '''
        Step through each column and get input from user
        OR Process a single column
        '''
        if single_col != 'no column specified':
            assert single_col in list(self.df), "column name not in dataframe (Did you misspell?)"
            loop_list = list(enumerate([single_col]))
            print("Column: {} will be updated\n".format(single_col))
            for list_check in [self.skip_cols, self.id_cols, self.float_cols, self.cat_cols, 
                               self.ordinal_cols, self.other_cols, self.binary_cols, self.response_cols]:
                if single_col in list_check:
                    list_check.remove(single_col)
                    #break
        else:
            loop_list = list(enumerate(list(self.df)))
            
        
        for i,c in loop_list:
            clear_output(wait=True)
            print("_____________________________")
            print("Pre-processing selection\n")
            DataPreProcess.fun_progress_in_loop( i, len(loop_list) )
            print("\n\n*** {} *** \n\n".format(c))
            
            
            

            temp_col = self.df[c].dropna().values
            num_missing = self.df[c].shape[0] - temp_col.shape[0]
            percent_missing = num_missing / self.df[c].shape[0]
            if percent_missing <= 0.05:
                percent_color_text = '\x1b[5;30;42m'
            elif percent_missing <= 0.25:
                percent_color_text = '\x1b[2;40;33m'
            else:
                percent_color_text = '\x1b[0;37;41m'


            print(self.df[c].describe())
            print(percent_color_text + "{:,} ({:.0%}) number of NAs\x1b[0m".format(num_missing, percent_missing))
            sleep(0.5)
            see_hist = input("\nDo you want to see the distribution? (yes/[no]) \n") or "no"
            if see_hist == "yes":
                plt.hist(temp_col, bins=40)
                plt.xticks(rotation=45, ha="right")
                plt.title(c)
                plt.show()

            feat_selection = "none"
            while feat_selection not in self.feature_options:
                feat_selection = input("\nWhat type of data should this be? Options are: {}\n".format(self.feature_options)) or "skip"
                if feat_selection not in self.feature_options:
                    print("Please choose a valid option...")

            if feat_selection == 'skip':
                print("{} will be skipped".format(c))
                self.skip_cols.append(c)
                continue

            elif feat_selection == 'id':
                print("{} will be saved as ID for later".format(c))
                self.id_cols.append(c)
                continue

            elif feat_selection == 'response':
                print("{} will be saved as the RESPONSE variable for later".format(c))
                response_type = input("\nHow should this response variable be treated?\n['keep float', 'keep string', 'one-hot encode']\n")
                self.response_dict[c] = response_type
                self.response_cols.append(c)
                continue

            elif feat_selection == 'other':
                print("{} will be included in 'other' for review later".format(c))
                self.other_cols.append(c)
                continue

            elif feat_selection == 'ordinal':
                print("{} will be used as an ordinal feature".format(c))
                self.ordinal_cols.append(c)
                continue

            elif feat_selection == 'category':
                print("{} will be used as a categorical feature".format(c))
                self.cat_cols.append(c)
                continue

            elif feat_selection == 'binary':
                print("{} will be used as a binary feature".format(c))
                self.binary_cols.append(c)
                continue

            elif feat_selection == 'float':
                self.float_cols.append(c)
                skew_check = input("{} is floating point, do you want to check transformations? (yes/[no]) ".format(c)) or "no"
                plt.close()
                if skew_check == 'yes':
                    best_float_trans = DataPreProcess.recommend_float_transformation(temp_col)


                skew_decision = input("Which transformation do you want to make? ('yeojohnson', 'box-cox', 'log1p', 'square root', [none]) ") or "none"
                self.skew_transformer[c] = skew_decision
                continue

        print("\nDONE :)\n***********")
                                      
                                      
    def create_pipeline(self, cap_values = [0.10, 0.90], rare_group_cutoff = 0.10):
        '''
        Create pipeline based on input from process_columns method
        '''
                                      
        cap_dict = dict([(j, cap_values) for j in self.float_cols])
        yj_cols = [c for c,j in self.skew_transformer.items() if j=='yeojohnson']
        box_cols = [c for c,j in self.skew_transformer.items() if j=='box-cox']
        log_cols = [c for c,j in self.skew_transformer.items() if j=='log1p']
        sq_cols = [c for c,j in self.skew_transformer.items() if j=='square root']
        
        
        self.pipeline2 = Pipeline(steps = [], verbose = True)
        
        if self.float_cols != []:
            self.pipeline2.steps.append(
                (
                    "numeric_capping",
                    CappingTransformer(
                        quantiles = cap_dict,
                        verbose = False
                    )
                )
            )
            
            self.pipeline2.steps.append(
                (
                    "numeric_imputation",
                    MeanImputer(
                        columns = self.float_cols + self.ordinal_cols + self.binary_cols
                    )
                )
            )
            
        if self.cat_cols != []:
            self.pipeline2.steps.append(
                (
                    "category_null_value",
                    ArbitraryImputer(
                        impute_value = 'missing',
                        columns = self.cat_cols
                    ) 
                )
            )
            self.pipeline2.steps.append(
                (
                    "rare_level_grouping",
                    GroupRareLevelsTransformer(
                        columns = self.cat_cols, 
                        cut_off_percent = rare_group_cutoff, 
                        verbose = False
                    )
                )
            )
            
            cat_selection = input("\nHow do you want to treat category variables? Options are:\n{}\n".format(['One-hot encoding', 'leave as string']))
            if cat_selection == 'One-hot encoding':
                self.pipeline2.steps.append(
                    (
                        "one_hot_encoding",
                        OneHotEncodingTransformer(
                            columns = self.cat_cols, 
                            drop_original = True, 
                            verbose = False
                        )
                    )
                )
            else:
                self.cat_decision = 'category'   #NominalToIntegerTransformer
                self.pipeline2.steps.append(
                    (
                        "nominal_encode",
                        NominalToIntegerTransformer(
                            columns = self.cat_cols, 
                            verbose = False
                        )
                    )
                )
                
            
        if yj_cols + box_cols + log_cols + sq_cols != []:
            col_trans_ls = []
            if yj_cols != []:
                col_trans_ls.append(("yj", PowerTransformer(method='yeo-johnson'), yj_cols))  
            if box_cols != []:
                col_trans_ls.append(("box", PowerTransformer(method='box-cox'), box_cols))
            if log_cols != []:
                col_trans_ls.append(("log1p", FunctionTransformer(np.log1p), log_cols))
            if sq_cols != []:
                col_trans_ls.append(("sq root", FunctionTransformer(np.sqrt), sq_cols))
                
            ct = ColumnTransformer(col_trans_ls, remainder='passthrough', 
                                   sparse_threshold=0, verbose_feature_names_out=False)
            ct.set_output(transform='pandas')
            self.pipeline2.steps.append(
                ("float transformers", ct)
            )
            
        ct_mm = ColumnTransformer([('min_max_ct', MinMaxScaler(), self.float_cols + self.ordinal_cols)],
                                  remainder='passthrough', sparse_threshold=0, verbose_feature_names_out=False)
        ct_mm.set_output(transform='pandas')
            
        self.pipeline2.steps.append(
            ( "minmax", ct_mm )
        )
        
    def add_model(self, model):
        '''
        Add a model to the pipeline 
        '''
        if self.cat_decision == 'category':
            model.set_params(categorical_features = self.cat_cols)
        
        self.pipeline2.steps.append(
            ("my_model", model)
        )
        self.model_flag = 'model added'
                
        
    def fit_pipeline(self, df='no df', y='no y'):
        '''
        fit the pipeline with the columns that will be kept
        '''
        if type(df) == str:
            df = self.df.copy()
            
            
        if type(y)==str:    
            self.pipeline2.fit(df[self.float_cols + self.cat_cols + self.ordinal_cols + self.binary_cols])
        else:
            self.pipeline2.fit(df[self.float_cols + self.cat_cols + self.ordinal_cols + self.binary_cols], y)
        
    
    def return_pipeline(self):
        '''
        return the created pipeline
        '''
        return self.pipeline2
    
    
    def transform_new_data(self, new_df):
        '''
        return transformed new data - columns names need to match
        as_df = True. if true return as Pandas dataframe, otherwise return as numpy array
        '''
        if self.model_flag == 'model added':
            X_prep = self.pipeline2[:-1].transform(new_df[self.float_cols + self.cat_cols + self.ordinal_cols + self.binary_cols].copy())
            #try:
            #    return pd.DataFrame(X_prep, columns=self.pipeline2[-2].get_feature_names_out())
            #except:
            #    return pd.DataFrame(X_prep, columns=self.pipeline2[-1].get_feature_names_out())
            return X_prep
        else:
            return self.pipeline2.transform(new_df[self.float_cols + self.cat_cols + self.ordinal_cols + self.binary_cols])
        
        
    def return_response_values(self, df='no df'):
        '''
        return a numpy array of the response column
        TODO: make the transformation as noted in the dictionary
        '''
        if type(df) == str:
            df = self.df
            
        if self.response_cols == []:
            print("NO RESPONSE COLUMN SPECIFIED, CAN'T RETURN ANYTHING")
        else:
            return df[self.response_cols[0]].values
        
        
    def recommend_float_transformation(X):
        '''
        return the recommended best transformation (between yeo-johnson, square root, log1p, box-cox)
        '''
        min_x = np.min(X)
        original_skew = skew(X)
        yj_x = PowerTransformer(method='yeo-johnson').fit_transform(X.reshape(-1, 1))
        yj_skew = skew(yj_x)
        
        skew_list = [('No transformation', original_skew, abs(original_skew)), 
                     ('yeo-johnson', yj_skew, abs(yj_skew))]
        if min_x==0:
            fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, figsize=(10, 10))
        elif min_x>0:
            fig, ((axs1, axs2), (axs3, axs4), (axs5, axs6)) = plt.subplots(3, 2, figsize=(10, 10))
        else:
            fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(10, 6))
            
        axs1.hist(X, bins=40)
        axs1.set_title("No Transformation (skew: {})".format(original_skew))
        axs2.hist(yj_x, color='orange', bins=40)
        axs2.set_title("Yeo-Johnson (skew: {})".format(yj_skew))
        
        if min_x > 0:
            box_x = PowerTransformer(method='box-cox').fit_transform(X.reshape(-1, 1))
            box_skew = skew(box_x)
            skew_list.append(('box-cox', box_skew, abs(box_skew)))
            axs5.hist(box_x, color='yellow', bins=40)
            axs5.set_title("Box-Cox (skew: {})".format(box_skew))
        
        if min_x >= 0:
            log_x = np.log1p(X)
            log_skew = skew(log_x)
            skew_list.append(('log1p', log_skew, abs(log_skew)))
            axs3.hist(log_x, color='red', bins=40)
            axs3.set_title("Log1p (skew: {})".format(log_skew))
            
            sq_x = np.sqrt(X)
            sq_skew = skew(sq_x)
            skew_list.append(('sqrt', sq_skew, abs(sq_skew)))
            axs4.hist(sq_x, color='green', bins=40)
            axs4.set_title("Square root (skew: {})".format(sq_skew))
            
        best_solution = sorted(skew_list, key=lambda x: x[2])[0]    

        #fig.suptitle("Transformations for {}".format(c))
        plt.show()
        
        print("Best option is: {}, which has skew of {}\n".format(best_solution[0], best_solution[1]))
        return best_solution[0]
            
            
        
        
        
        