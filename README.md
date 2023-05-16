# data_preprocess_helper
Step through each feature to determine what preprocessing should be done. Then automatically create and fit a pipeline. Processing pipeline is based on sklearn and includes transformers from the [tubular](https://pypi.org/project/tubular/) package

----------

TO-DO:
1. ~~simplify class to not retain a copy of the dataframe~~ Done 5/12/2023. Now user has to pass in dataframe as a parameter (makes saving the preprocessor in a pickle file less bloated)
2. ~~allow for user to specify capping~~ Updated 5/12/2023. User can go through each specific feature to see what different capping values, or select quantiles for all
3. update response variable to be used for categorical encoding (one-hot, ordinal encoding, etc.)
4. Create a version that only uses sklearn preprocessing methods (in case tubular is unavailable)
