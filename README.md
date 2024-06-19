# Kronecker Generative Models for Power Law Patterns in Real-World Hypergraphs

We provide datasets and source code for (1) **Discoveries** of log-logistic or power-law patterns in real-world hypergraphs and (2) **HyperK** which is a hypergraph generator based on kronecker product and **SingFit** for estimating initiator

```
# File Organization
|__ Property/                   
    |__ dataset/                # Hypergraphs: i-th line indicates the nodes included in the i-th hypergraph 
    |__ rawdata/                # Location of raw data downloaded from https://www.cs.cornell.edu/~arb/data/
    |__ analyze_data/           # Used for analyzing discoveries in real-world hypergraphs
    |__ analyze_fit/            # Used for evaluating hypergraph generators' fit to real-world hypergraphs
    |__ analyze_extrapolation/  # Used for evaluating hypergraph generators' extrapolation of real-world hypergraphs
    |__ src/                    # Source code for computing nine properties
    |__ run/                    # Command files for computing nine properties and evaluating generators
|
|__ Model/                      
    |__ input/                  # Real-world hypergraphs in a different format: each line indicates {column index} {row index} {value} of the incidence matrix
        |__ svdata/             # Pre-computed singular values from real-world hypergraphs
    |__ run/                    # Used for training HyperK in search spaces
    |__ run_full/               # Used for generating hypergraphs from trained HyperK
    |__ run_half/               # Used for extrapolating hypergraphs from trained HyperK
    |__ *.py                    # Source code for HyperK and SingFit
    |__ saved_model.tar.gz      # Best-performing models for fitting and extrapolating
```

## (0) Data Preprocessing
The datasets are downloaded from [here](https://www.cs.cornell.edu/~arb/data/). After placing these datasets in the `Property/rawdata/` directory, we preprocess them using the scripts in `Property/dataset/`:

* `preprocess.py`: Preprocesses hypergraphs and saves them in the Property/dataset/ directory. These are used for computing nine properties.
* `preprocess_half.py`: Preprocesses hypergraphs by including only time-sorted hyperedges until the number of nodes is half of the original. These are saved in the Property/dataset/ directory for the extrapolation task.
* `preprocess_kron.py`: Reads preprocessed hypergraphs and saves them in a different format in the Model/input/ directory.

Even though you can obtain all the datasets we used by following the provided preprocessing code, we also provide the complete training datasets for both fit and extrapolation from eleven real-world hypergraphs in the `Model/input/` directory. Additionally, we include the real-world hypergraph and generated ones via five baselines using the email-Eu dataset as an example.


## (1) Discoveries

You can analyze *eight* discoveries:
* D1. Power-law distributions in node pair degrees, intersection sizes, and singular values.
* D2. Log-logistic distributions in node degrees and hyperedge sizes.
* D3. Power-law patterns in clustering coefficients, egonet density, and overlapness.

By following the script `Property/run/run.sh`, eight properties are computed, and the outputs will be saved in the `Property/results/answer/[data name]` directory.

```
cd Property/
make
./bin/Evaluation --inputpath ./dataset/${data name} --outputdir ./results/answer/${data name}/ --dupflag
cd src
python calculation_helper.py --inputpath ../dataset/${data name} --outputdir ../results/answer/${data name}/ --sv --dupflag
python powerlaw_test.py --dataname ${data name} --test_type pairdeg
python powerlaw_test.py --dataname ${data name} --test_type intersection
python powerlaw_test.py --dataname ${data name} --test_type sv
```

You can then analyze these properties visually and statistically:

* `Property/analyze_data/plot_discoveries.py`: Plots eight properties in log-log scale and checks the uniformity of slopes within the same domain.
* `Property/analyze_data/stats.py`: Checks log-likelihood ratios for fitting to power-law distributions, and computes R² scores and slopes of linear regression.


## (2) HyperK and SingFit

We provide source code for training HyperK using SingFit

### How to Train HyperK

You can **train** HyperK by following the instructions in `Model/run/run_{half, full}.sh`,
```
python main_sv.py train --dataset {data name}
                        --device {cuda number}
                        --numparam {parameter count constraint}
                        --lr {learning rate}
                        --annealrate {annealing rate for tuning temperature in gumbel softmax}
                        --num_tie {number of tie in training}
                        --sizelambda {lambda for size loss}
                        --deglambda {lambda for degree loss}
```
To **generate (or extrapolate)** a hypergraph using the trained HyperK, follow the instructions in `Model/run_full/run_gen.sh` (or `Model/run_half/run_extrapolate.sh`):

```
python main.py eval --dataset {target data name}
                    --device {cuda number}
                    --load_path {trained model path}
                    --save_path {save path for outputs}
                    --save_iter {number of generating}
                    --extflag # used when extrapolating
```

### How to Evaluate HyperK

#### Fitting to Real-world Hypergraphs

You can compute nine properties from the hypergraphs generated by HyperK and evaluate them against baselines by following these steps:
1. Save Hypergraphs in Proper Format. Run the following command to read the generated hypergraphs and save them in the proper format for computing properties
```
cd Property/results/
python make_input_list.py --dataname {target_dataname} --inputdir {directory_path_for_trained_HyperK}
```
2. Compute Properties. Run the following script to compute nine properties from all hypergraphs generated by HyperK:
```
cd Property/run/
./run_eval_HyperK_{target_dataname}.sh
```
3. Select Best Hypergraph. Run the following command to select the best generated hypergraph:
```
cd Property/analyze_fit/
python ablation.py --dataname {target_dataname} --ablation_target HyperK
```
4. Prepare Competitor Results. Compute nine properties from hypergraphs generated by competitors (HyperCL, HyperFF, HyperLAP, HyperPA, THera):
```
cd Property/run/
./run_{cl, ff, lap, pa, tr}.sh
# For HyperFF and THera, select the best hyperparameters in the search spaces
cd Property/analyze_fit/
python ablation.py --dataname {target_dataname} --ablation_target {hyperff or thera}
```
5. Generate Comparison Table. Run the following command to save the comparison of fitting to the target hypergraph in nine properties:
```
cd Property/analyze_fit/
# generate table
python gen_table.py --dataname {target_dataname} --outputdir {output_directory_for_result_csv}
# generate figures
python plot.py --dataname {target_dataname}
```

#### Extrapolating to Real-world Hypergraphs
You can also compute nine properties from the hypergraphs extrapolated by HyperK,
```
cd Property/results/hyperk/
python make_input_list_half.py --dataname {half_data_name} # make shell script for computing nine properties from all extrapolated hypergraphs in Model/result/{half_data_name}/full/*
cd ../../run
./run_hyperk_{half_data_name}.sh # run the script for computing properties

cd Property/results/hyperk_half/
python make_input_list.py --dataname {half_data_name} # make shell script for computing nine properties from all half-sized hypergraphs in Model/result/{half_data_name}/*
cd ../../run
./run_hyperk_{half_data_name}-half.sh # run the script for computing properties
```


You can compare the distributions from generators by `Property/analyze/Compare Figure {Extrapolation}.ipynb`.

You can also evaluate the generators by rankings and z-scores by `Property/analyze/Compare Table {Extrapolation}.ipynb`.


- - -

## Pre-trained Model

You can download best-performing models by downloading `Model/saved_model.tar.gz` through git LFS.
The configurations of these trained models are as follows:

* Fitting to Real-world Hypergraphs

| Data | (N_1, M_1) | K | Learning Rate | Number of Tie | Size Lambda | Degree Lambda |
| --- | --- | --- | --- | --- | --- | --- | 
| email-Enron | (3, 5) | 6 | 0.010 | 2 | 0.01 | 0.0010 |
| email-Eu | (4, 8) | 6 | 0.008 | 3 | 0.00 | 0.0010 |
| contact-high | (4, 12) | 5 | 0.010 | 2 | 1.50 | 0.0000 |
| contact-primary | (4, 19) | 4 | 0.010 | 2 | 0.60 | 0.0001 |
| NDC-classes | (5, 9) | 5 | 0.008 | 2 | 0.10 | 0.0000 |
| NDC-substances | (18, 49) | 3 | 0.005 | 2 | 0.50 | 0.0010 |
| tags-ubuntu | (8, 23) | 4 | 0.010 | 2 | 2.00 | 0.0100 |
| tags-math | (7, 31) | 4 | 0.005 | 2 | 1.00 | 0.0010 |
| threads-ubuntu | (8, 8) | 6 | 0.010 | 2 | 0.50 | 0.1000 |
| threads-math | (12, 15) | 5 | 0.003 | 2 | 1.00 | 0.0100 |
| coauth-geology | (6, 6) | 8 | 0.005 | 2 | 0.00 | 0.0010 |

* Extrapolating Real-world Hypergraphs
  
| Data | (N_1, M_1) | K | Learning Rate | Number of Tie | Size Lambda | Degree Lambda |
| --- | --- | --- | --- | --- | --- | --- | 
| email-Enron-half | (3, 4) | 5 | 0.010 | 2 | 1.00 | 0.0010 |
| email-Eu-half | (4, 8) | 5 | 0.010 | 2 | 1.50 | 0.0010 |
| contact-high-half | (4, 12) | 4 | 0.010 | 2 | 1.00 | 0.0010 |
| contact-primary-half | (4, 18) | 4 | 0.010 | 2 | 1.00 | 0.0000 |
| NDC-classes-half | (5, 9) | 4 | 0.001 | 2 | 1.00 | 0.0001 |
| NDC-substances-half | (5, 8) | 5 | 0.010 | 2 | 0.01 | 0.0010 |
| tags-ubuntu-half | (3, 4) | 7 | 0.010 | 2 | 1.00 | 0.0100 |
| tags-math-half | (4, 8) | 5 | 0.010 | 2 | 1.00 | 0.0100 |
| threads-ubuntu-half | (5, 5) | 7 | 0.001 | 2 | 0.50 | 2.0000 |
| threads-math-half | (18, 22) | 4 | 0.010 | 2 | 1.00 | 0.5000 |
| coauth-geology-half | (7, 7) | 7 | 0.100 | 2 | 0.00 | 0.0000 |


## Environment

The environment for running the codes is specified in `requirements.txt`
We use RTX2080Ti and AMD Ryzen 7 3700X.
