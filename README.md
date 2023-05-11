# data_preprocess_helper
Step through each feature to determine what preprocessing should be done. Then automatically create and fit a pipeline. Processing pipeline is based on sklearn and includes transformers from the [tubular](https://pypi.org/project/tubular/) package

----------

TO-DO:
1. simplify class to not retain a copy of the dataframe
2. allow for user to specify capping
3. update response variable to be used for categorical encoding (one-hot, ordinal encoding, etc.)
