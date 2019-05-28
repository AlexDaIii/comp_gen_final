# EN 601.448 Computational Genomics Final Project

Elysia Chou (echou4), Keefer Chern (kchern1), Alexander Chang (achang56)
May 1, 2019

For our code, we have two separate subprojects:

- GCP analysis
- Gene expression analysis

## Data

The preprocessed data files that are less than 100MB is stored in `./data` directory. The preprocessed
data files that are included are listed below:

    expr_labels.csv
    features.csv
    labels.csv

The rest of the processed data and the raw, unprocedded data can be found at this link
[here](https://drive.google.com/drive/folders/1NihIZKITpMfff9118KInw3QksIczCojV?usp=sharing)

We obtained the original raw data files from the CCLE website
[here](https://portals.broadinstitute.org/ccle/data). After running the preprocessing
bash scripts, it will also include the CSV files with extracted data that we used as the data
for our project.

## How to run

The python file `runner.py` can call all of the preprocessing and model files. The 
script has the following arguments:

    --preprocess_gcp      Adding this argument will pre-process gcp data
    --preprocess_gene     Adding this argument will pre-process gene data
    --run_histone         Adding this argument will run elastic net and analysis
                          on histone data
    --run_gene            Adding this argument will run elastic net and analysis
                          on gene data
    --run_pca             Adding this argument will run pca on histone data

    Required if running GCP preprocessing:
    --gcp_X               The unprocessed gcp data.
    --gcp_Y               The unprocessed gcp labels.
    --gcp_outputY         The csv file to output the gcp/histone labels
    --gcp_outputX         The csv file to output the gcp/histone  features

    Required if running Gene preprocessing:
    --gene_X              The unprocessed gene data.
    --gene_Y              The unprocessed gene labels.
    --gene_outputY        The csv file to output the gene labels
    --gene_outputX        The csv file to output the gene features

    Required if running gcp/histone analysis and PCA:
    --histone_features    The gcp/histone data to use for training.
    --histone_labels      The gcp/histone labels to use for training

    Required if running gene analysis:
    --gene_features       The gene data to use for training.
    --gene_labels         The gene labels to use for training
    Optional arguments for running gene analysis:
    --hyperparam          Adding this argument will run the hyperparameter
                          search on gene expr data
    --load_new            Adding this argument will load the OLD format gene
                          expr data - the processed data found in the Google Drive
                          expr_features.csv and expr_labels.csv

    Required for both running gcp/histone analysis and PCA and gene analysis:
    --output_dir          The output directory for the models

    Optional argument for timing how long each script takes
    --time                Adding this argument will time the runtimes`

You can also call the bash script `run.sh` which is set up to call the `runner.py`
script to perform the histone and gene expression elastic net regression and analysis
and perform PCA on the histone features. It does not call `runner.py` to
perform the preprocessing.

We also have individual shell files, one for each subproject. Our structure of our
zipped code is indicated below. For more details, please see our methods section
of our final report.

### Preprocessing Files

These are stored in the `./preprocess` folder. These files preprocess the original raw data into
the features and label data used for the models and analysis.

#### GCP Data Preprocessing

This is stored in the `./preprecess/GCP` folder.

`run_GCP.sh`:  
  Bash script to extract GCP data from the original raw data. Runs:

      extractData.py -  extract relevant data from raw data. Outputs features.CSV
                    and labels.csv to the /data directory.

#### Gene Data

This is stored in the `./preprecess/Gene_Expr` folder.

`run_expr.sh`:
  Bash script to extract genetic data from the original raw data. Runs:

    extractData_expr.py - extract relevant data from raw data. Outputs
                          expr_features.CSV and expr_labels.csv to the /data
                          directory. Displays a DtypeWarning but that is for the
                          first column, which the cancer cell line column.

### Analysis and Models

These are stored in the `./model` folder. These files create the Elastic Model and analyze the data.

#### Histone

`histone.sh`:
  Bash script for GCP analysis. Runs:

    histone.py -  preprocesses data, runs 10-fold cross-validation for
                  hyperparameter tuning, computes the Spearman correlation, and
                  run linear regression on each of the 39 features individually
                  with Benjamini Hochberg multiple hypothesis correction.
                  Outputs 8 files to the /output directory:
                  00 IC50: 10 smallest MSE values for hyperparameter tuning
                  01 IC50: 10 largest R^2 values for hyperparameter tuning
                  02 Amax: 10 smallest MSE values for hyperparameter tuning
                  03 Amax: 10 largest R^2 values for hyperparameter tuning
                  04 ActArea: 10 smallest MSE values for hyperparameter tuning
                  05 ActArea: 10 largest R^2 values for hyperparameter tuning
                  06 Spearman correlation heatmap across 39 features
                  07 # of significant p-values for individual linear regression

#### Gene

`gene.sh`:
  Bash script for gene expression analysis. Runs:
    gene.py - preprocesses, runs 10-fold cross-validation for hyperparameter
              tuning, bootstrapping with manually pre-determined alpha and
              lambda values, determines MSE for test set. Outputs 8 files to
              the /output directory:
              08 IC50: 10 smallest MSE values for hyperparameter tuning
              09 IC50: 10 largest R^2 values for hyperparameter tuning
              10 Amax: 10 smallest MSE values for hyperparameter tuning
              11 Amax: 10 largest R^2 values for hyperparameter tuning
              12 ActArea: 10 smallest MSE values for hyperparameter tuning
              13 ActArea: 10 largest R^2 values for hyperparameter tuning
              14 Top 10 genes with highest bootstrapped frequency and
                 corresponding weights for activity area (ActArea)
              15 Train MSE and test MSE for bootstrapped elastic net model

#### PCA

`pca.sh`:
  Bash script for performing PCA on the histone data. Runs:

    gene.py - preprocesses and performs PCA on the unstandardized data.
              Graphs the top 10 principal components and the features
              embedded in the first top 2 PCs. Outputs 2 files to
              the /output directory:
              21 Top 10 PC: Bar graph of the top 10 PCs
              22 PC2 vs PC1: Plot of the top 2 PCs

## Output

`./output` contains all our output files created after the bash scripts are run. The order
of the files is maintained through an index (00, 01, 02, ...) that indicates the
order in which the outputs are meant to be read.
