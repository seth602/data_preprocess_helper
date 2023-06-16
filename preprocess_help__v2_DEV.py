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

#######################################################################################
####### Purpose of this class is to help in creating data transformation 
####### pipelines. The user can either step through each column one at a time,
####### or use the recommended pipeline which tries to make the best decisions
####### based on statistical tests and best practices.
#######################################################################################


class DataPreProcess:
    def __init__(self) -> None:
        '''
        Initialize DataPreProcess class. The pandas DF is not passed yet,
        But dictionaries will be created to keep track of each column's transformation
        '''
        self.column_type = {}
        self.column_transformation = {}
        self.numeric_qunatiles = {}
        self.feature_options = ['skip', 'id', 'response', 'float', 'category', 'binary', 'ordinal', 'other']
        self.pipeline2 = Pipeline(steps = [], verbose = True)
        
    def fun_progress_in_loop(i_iter: int, len_iter: int) -> None:
        '''
        Make a green progress bar while iterating through the columns.
        This isnt necessary, but i did it just for fun (and it looks kinda cool)
        '''
        percent_color_text = '\x1b[5;37;42m'
        progress_bar = (i_iter+1)/len_iter
        num_green_spaces = " " * int(progress_bar * 50)
        num_remaining_space = " " * (50 - int(progress_bar * 50))
        percent_left = str( int(progress_bar*100) )
        print(percent_left + "%\t[" + percent_color_text + num_green_spaces + "\x1b[0m" + 
              num_remaining_space + "] {}/{} columns".format((i_iter+1), len_iter))

    def input_with_options(input_text: str, input_options: list, default_option: str) -> str:
        '''
        if valid option isnt given, ask again
        '''
        user_input = None
        while user_input not in input_options:
            user_input = input(input_text + f" [{default_option}]\n\nOptions: {input_options}") or default_option

        return user_input

    def select_caps(X: pd.Series, col: str) -> list:
        '''
        Allow user to see the distribution with different percent cutoffs and select quantile caps

        Return: [lower_quantile, upper_quantile]
        '''
        
        clear_output(wait=True)
        temp_vals = X.dropna().values
        #min_val = np.min(temp_vals)
        #max_val = np.max(temp_vals)
        bot_5 = np.percentile(temp_vals, 5)
        bot_10 = np.percentile(temp_vals, 10)
        top_95 = np.percentile(temp_vals, 95)
        top_90 = np.percentile(temp_vals, 90)

        plt.hist(temp_vals, bins=40)
        plt.axvline(x=bot_5, color='b', ls='--', label='5th percentile: {}'.format(bot_5))
        plt.axvline(x=bot_10, color='r', ls='-', label='10th percentile: {}'.format(bot_10))
        plt.axvline(x=top_90, color='purple', ls='-.', label='90th percentile: {}'.format(top_90))
        plt.axvline(x=top_95, color='orange', ls=':', label='95th percentile: {}'.format(top_95))
        plt.xticks(rotation=45, ha="right")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.title(col)
        plt.show()
        cap_decision_lower = float(input("Input lower cap (percent ONLY): "))
        cap_decision_upper = float(input("Input upper cap (percent ONLY): "))
        assert cap_decision_lower < cap_decision_upper, "upper bound can't be smaller than lower bound"
        assert (cap_decision_lower >= 0) & (cap_decision_lower <= 1) & (cap_decision_upper >= 0) & (cap_decision_upper <= 1), "bounds need to be between [0, 1]"

        return [cap_decision_lower, cap_decision_upper]



    def recommend_float_transformation(X: pd.Series, view_plots: bool = True) -> tuple:
        '''
        return the recommended best transformation (between yeo-johnson, square root, log1p, box-cox)
        '''
        X = X.dropna().values
        min_x = np.min(X)
        original_skew = skew(X)
        yj_x = PowerTransformer(method='yeo-johnson').fit_transform(X.reshape(-1, 1))
        yj_skew = skew(yj_x)
        
        skew_list = [('No transformation', original_skew, abs(original_skew)), 
                     ('yeo-johnson', yj_skew, abs(yj_skew))]
        
        if min_x > 0:
            box_x = PowerTransformer(method='box-cox').fit_transform(X.reshape(-1, 1))
            box_skew = skew(box_x)
            skew_list.append(('box-cox', box_skew, abs(box_skew)))
        
        if min_x >= 0:
            log_x = np.log1p(X)
            log_skew = skew(log_x)
            skew_list.append(('log1p', log_skew, abs(log_skew)))
            
            sq_x = np.sqrt(X)
            sq_skew = skew(sq_x)
            skew_list.append(('sqrt', sq_skew, abs(sq_skew)))
            
        best_solution = sorted(skew_list, key=lambda x: x[2])[0]    

        #fig.suptitle("Transformations for {}".format(c))
        if view_plots:
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
                axs5.hist(box_x, color='yellow', bins=40)
                axs5.set_title("Box-Cox (skew: {})".format(box_skew))
            if min_x >= 0:
                axs3.hist(log_x, color='red', bins=40)
                axs3.set_title("Log1p (skew: {})".format(log_skew))
                axs4.hist(sq_x, color='green', bins=40)
                axs4.set_title("Square root (skew: {})".format(sq_skew))

            plt.show()
        
        print("Best option is: {}, which has skew of {}\n".format(best_solution[0], best_solution[1]))
        return best_solution[0], [s[0] for s in skew_list]



    # MAKE A FUNCTION TO PROCESS EACH TYPE OF VARIABLE
    def process_categorical_column(self, col_name: str, cat_output: str, auto_process: bool = False) -> None:
        '''
        for a chosen categorical column, either get user's input (if auto_process == 'no')
        or decide best transformations
        '''
        list_of_transformers = []
        if auto_process == False:
            if cat_output is None:
                text_for_input = f"\nHow should {col_name} be handled?"
                list_for_input = ['one-hot encode', 'nominal encode', 'category type']
                default_for_input = 'one-hot encode'
                cat_output = DataPreProcess.input_with_options(text_for_input, list_for_input, default_for_input)

            rare_level_decision = DataPreProcess.input_with_options("Do you want to perform rare level grouping?", 
                                                                    ["yes", "no"], "yes")

            impute_decision = DataPreProcess.input_with_options("Do you want to impute missing values?", 
                                                                ["yes", "no"], "yes")

        else:
            rare_level_decision = "yes"
            impute_decision = "yes"

        if impute_decision == "yes":
            list_of_transformers.append("impute category")
        if rare_level_decision == "yes":
            list_of_transformers.append("rare level category grouping")
        list_of_transformers.append(cat_output)

        self.column_transformation[col_name] = list_of_transformers


    def process_float_column(self, col_values: pd.Series, col_name: str, quantile_ls: list, auto_process: bool = False) -> None:
        '''
        for a chosen floating point column, either get user's input (if auto_process == 'no')
        or decide best transformations and quantile capping
        '''
        list_of_transformers = []
        ###############
        # skew transformers
        ################
        if auto_process == False:
            view_transformations = DataPreProcess.input_with_options("Do you want to view data transformations?", 
                                                                    ["yes", "no"], "yes")
            if view_transformations == "yes":
                best_transformation, ls_options = DataPreProcess.recommend_float_transformation(col_values, view_plots=True)
            else:
                best_transformation, ls_options = DataPreProcess.recommend_float_transformation(col_values, view_plots=False)    

            text_for_trans_input = f"What transformation do you want for {col_name}?"
            transformation_decision = DataPreProcess.input_with_options(text_for_trans_input, ls_options, best_transformation)

        else:
            transformation_decision, _ = DataPreProcess.recommend_float_transformation(col_values, view_plots=False) 

        ###############
        # quantile decision
        ################
        if auto_process == False:
            set_quantiles = DataPreProcess.input_with_options(f"Do you want to set caps on {col_name}?", 
                                                                    ["yes", "no"], "yes")
            if set_quantiles == "yes":
                quantile_ls = DataPreProcess.select_caps(col_values, col_name)
                self.numeric_qunatiles[col_name] = quantile_ls

        else:
            set_quantiles = "yes"
            self.numeric_qunatiles[col_name] = quantile_ls

        ###############
        # mean impute decision
        ################
        if auto_process == False:
            impute_decision = DataPreProcess.input_with_options(f"Do you want to impute missing values with the mean?", 
                                                                    ["yes", "no"], "yes")
        else:
            impute_decision = "yes"

        # put it all together
        if set_quantiles == "yes":
            list_of_transformers.append("cap")
        if transformation_decision != "No transformation":
            list_of_transformers.append("skew transform")
        if impute_decision == "yes":
            list_of_transformers.append("impute")

        self.column_transformation[col_name] = list_of_transformers


    def determine_col_type(self, X: pd.Series, col_type: str = None) -> None:
        '''
        Take in a column (pd.Series) and determine the best column type
        Update the self.column_type dictionary
        '''
        from pandas.api.types import is_string_dtype
        from pandas.api.types import is_numeric_dtype

        if col_type is None:
            if is_string_dtype(X):
                uniq_percent = len(set(X)) / len(X)
                if uniq_percent < 0.05:
                    self.column_type[X.name] = "category"
                else: 
                    self.column_type[X.name] = "skip"

            if is_numeric_dtype(X):
                if len(set(X)) == 2:
                    self.column_type[X.name] = "binary"
                else:
                    self.column_type[X.name] = "numeric"

        else:
            self.column_type[X.name] = col_type


            

    def auto_process_columns(self, df: pd.DataFrame, single_col: str = None):
        '''
        Step through each column (or process single col) automatically and choose best option
        '''
        if single_col is not None:
            assert single_col in list(df), "column name not in dataframe (Did you misspell?)"
            loop_list = list(enumerate([single_col]))
        else:
            loop_list = list(enumerate(list(df)))
            
        
        for i,c in loop_list:
            clear_output(wait=True)
            print("_____________________________")
            print("Pre-processing selection\n")
            DataPreProcess.fun_progress_in_loop( i, len(loop_list) )
            print("\n\n*** {} *** \n\n".format(c))

            temp_col = df[c].dropna().values
            num_missing = df[c].shape[0] - temp_col.shape[0]
            percent_missing = num_missing / df[c].shape[0]
            if percent_missing <= 0.05:
                percent_color_text = '\x1b[5;30;42m'
            elif percent_missing <= 0.25:
                percent_color_text = '\x1b[2;40;33m'
            else:
                percent_color_text = '\x1b[0;37;41m'

            print(df[c].describe())
            print(percent_color_text + "{:,} ({:.0%}) number of NAs\x1b[0m".format(num_missing, percent_missing))
            sleep(0.5)
            if percent_missing > 0.25:
                self.column_type[c] = "skip"
                continue

            DataPreProcess.determine_col_type(self, df[c], col_type=None)

            if self.column_type[c] == 'numeric':
                DataPreProcess.process_float_column(self, df[c], col_name=c, quantile_ls=[0.05, 0.90], auto_process=True)

            elif self.column_type[c] == 'category':
                DataPreProcess.process_categorical_column(self, col_name=c, cat_output='one-hot encode', auto_process=True)

            elif self.column_type[c] == 'binary':
                self.column_transformation[c] = ["impute"]

    
        
        
    def process_columns(self, df: pd.DataFrame, single_col: str = None):
        '''
        Step through each column and get input from user
        OR Process a single column
        '''
        if single_col is not None:
            assert single_col in list(df), "column name not in dataframe (Did you misspell?)"
            loop_list = list(enumerate([single_col]))

        else:
            loop_list = list(enumerate(list(df)))
            
        
        for i,c in loop_list:
            clear_output(wait=True)
            print("_____________________________")
            print("Pre-processing selection\n")
            DataPreProcess.fun_progress_in_loop( i, len(loop_list) )
            print("\n\n*** {} *** \n\n".format(c))
            
            
            

            temp_col = df[c].dropna().values
            num_missing = df[c].shape[0] - temp_col.shape[0]
            percent_missing = num_missing / df[c].shape[0]
            if percent_missing <= 0.05:
                percent_color_text = '\x1b[5;30;42m'
            elif percent_missing <= 0.25:
                percent_color_text = '\x1b[2;40;33m'
            else:
                percent_color_text = '\x1b[0;37;41m'


            print(df[c].describe())
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





                                      
                                      
    def create_pipeline(self, df, cap_values = [0.10, 0.90], rare_group_cutoff = 0.10):
        '''
        Create pipeline based on input from process_columns method
        '''
                                      
        quant_dict = dict([(j, cap_values) for j in self.float_cols])
        val_dict = None
        yj_cols = [c for c,j in self.skew_transformer.items() if j=='yeojohnson']
        box_cols = [c for c,j in self.skew_transformer.items() if j=='box-cox']
        log_cols = [c for c,j in self.skew_transformer.items() if j=='log1p']
        sq_cols = [c for c,j in self.skew_transformer.items() if j=='square root']
        
        self.pipeline2 = Pipeline(steps = [], verbose = True)
        
        if self.float_cols != []:
            cap_decision = input("Do you want to cap numeric features?\n['yes']/'no'") or 'yes'
            if cap_decision == 'yes':
                pick_cap = input("Do you want to pick each feature's Caps? Otherwise stick with {}\n'yes'/'no'".format(cap_values))
                if pick_cap=='yes':
                    quant_dict, val_dict = DataPreProcess.select_caps(df, self.float_cols)

                if quant_dict is not None:    
                    self.pipeline2.steps.append(
                        (
                            "numeric_capping_quant",
                            CappingTransformer(
                                quantiles = quant_dict,
                                verbose = False
                            )
                        )
                    )
                if val_dict is not None:    
                    self.pipeline2.steps.append(
                        (
                            "numeric_capping_val",
                            CappingTransformer(
                                capping_values = val_dict,
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
            
            cat_selection = input("\nHow do you want to treat category variables? Options are:\n{}\n".format(['One-hot encoding', 'ordinal encode', 'leave as string']))
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
            elif cat_selection == 'ordinal encode':
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
                
        
    def fit_pipeline(self, df, y='no y'):
        '''
        fit the pipeline with the columns that will be kept
        '''
        if type(df) == str:
            df = df.copy()
            
            
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
        # TODO: no need to model_flag, just look to see if the name of the last pipeline step is 'my_model'
        if self.model_flag == 'model added':
            X_prep = self.pipeline2[:-1].transform(new_df[self.float_cols + self.cat_cols + self.ordinal_cols + self.binary_cols].copy())
            return X_prep
        else:
            return self.pipeline2.transform(new_df[self.float_cols + self.cat_cols + self.ordinal_cols + self.binary_cols])
        
        
    def return_response_values(self, df):
        '''
        return a numpy array of the response column
        TODO: make the transformation as noted in the dictionary
        '''            
        if self.response_cols == []:
            print("NO RESPONSE COLUMN SPECIFIED, CAN'T RETURN ANYTHING")
        else:
            return df[self.response_cols[0]].values
        
        
    
            
            
        
        
        
        