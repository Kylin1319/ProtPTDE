<h1 align="center">ProtPTDE</h1>

- **Documentation** [![Documentation badge](https://img.shields.io/readthedocs/tidymut/latest?logo=readthedocs&logoColor=white)](https://tidymut.readthedocs.io/en/): https://protptde-usage-guidelines.readthedocs.io/en/latest/

ProtPTDE (Protein Pre-Training Model-Assisted Protein Directed Evolution) is a computational strategy designed to assist protein directed evolution by integrating multiple deep learning models. A key design highlight of this framework lies in its streamlined parameter management: we have centralized the majority of parameters and hyperparameters involved in the entire architectural workflow and fitness prediction framework into a single configuration file, `config/config.json`.  

This centralized structure enables **unified parameter governance**: when users need to adjust a parameter, they only need to modify the target entry in `config.json`—the system will automatically sync this update across all associated scripts. This eliminates the cumbersome, error-prone process of manually searching through multiple files to modify parameters individually, thereby achieving highly centralized control and automated parameter tuning.  

Furthermore, we have prioritized **framework extensibility** to accommodate diverse research needs:  
1. **Adding custom protein language models**: Users can easily integrate new protein language models by developing their own `function.py` files, following the template provided in `generate_features/{model}_embedding/function.py`. Once created, the new model is automatically incorporated into the framework’s model search scope, requiring no extensive modifications to the core codebase.  
2. **Supporting multi-model embedding concatenation**: We have expanded the framework’s capability to concatenate embeddings from **multiple models** (instead of limiting to a single model). This design not only grants users greater flexibility in model selection but also allows leveraging richer, multi-source embedding information—enhancing the potential for predicting the structure and fitness of complex proteins.


## Install software on Linux

1. install `Anaconda` / `Miniconda` software

2. install Python packages

```bash
conda create -n Prot_PTDE python=3.13
conda activate Prot_PTDE

pip install torch==2.7.1
pip install tqdm
pip install click
pip install biopython
pip install "pandas[excel]"
conda install numba
pip install scikit-learn
pip install more-itertools
pip install iterative-stratification
pip install optuna
pip install transformers
pip install einops
pip install seaborn
pip install plotly
```

## Usage

* The model requires two essential input files:
A ``.xlsx`` file (referred to as mutation_data_file) stored in the ``data/`` directory, which contains mutation information and their corresponding fitness values.
A FASTA file of the wild-type sequence, named ``result.fasta``, located in the ``features/wt/`` directory.



* Bash scripts with filenames in the format of ``2000.sh``, ``2001.sh``, etc.—specifically those located in the ``01_cross_validation/`` and ``02_final_model/`` folders—must be renamed according to your server configuration. Note that the last digit of each such script's filename indicates the GPU card number used on the server (e.g., ``2000.sh`` corresponds to GPU card 0, ``2013.sh`` to GPU card 3, and so forth). Additionally, ensure the corresponding script names (e.g., ``2000.sh``, ``2001.sh``) referenced in ``01_train.sh`` (within both ``01_cross_validation/`` and ``02_final_model/``) are updated to match.


1. processing data and generate embeddings

Change the value of the key **basic_data_name** in the ``config/config.json`` to the name of the mutation_data_file (without the extension).

```bash
# processing data
cd data
python convert_xlsx_to_csv_and_generate_fasta_file.py
cd ../

# generate embeddings of all models(and it will record the embedding dimension for each corresponding model under the all_model key in the config/config.json)
cd generate_features
python generate_all_embeddings.py
cd ../
```

2. cross validation

Before concatenating multiple model embeddings, determine the output dimension of the linear transformation for single model and write it to the **single_model_embedding_output_dim** key in the ``config/config.json`` file.

Modify the desired configuration parameters under the **cross_validation** key in the ``config/config.json``. The **hyperparameter_search** field specifies the optional ranges for hyperparameter search using the Optuna library. The **model_number** field determines how many models you want to select for combination. The **training_parameter** field is used to set the basic parameters for model training.

```bash
cd 01_cross_validation
bash 01_train.sh
python 02_Dis_cross_validation.py
cd ../
```

Select the best hyperparameters from the displot (``Dis_cross_validation.pdf``) and write them in **best_hyperparameters** key in the ``config/config.json``. They are **selected_models**, **num_layer** and **max_lr**.

3. train and finetune

Adjust the number of model training runs and write it to the **ensemble_size** key in the ``config/config.json``. The training parameters are the same each time, except for model initialization and the batch order of training data provided by DataLoader. The results are saved independently and finally merged and analyzed to evaluate the stability of the final prediction results.

Modify the basic parameters for model training and finetuning under the **final_model** key in the ``config/config.json``.

```bash
cd 02_final_model
bash 01_train.sh
python 02_plot_random_seed_train.py
```

Select a good randomseed based on the scatter plot (``Scatter_best_train_test_epoch_ratio.html``) and write it to the **best_hyperparameters** key in the ``config/config.json``.

```bash
bash 03_train_ensemble.sh
bash 04_finetune.sh
bash 05_finetune_ensemble.sh
cd ../
```

4. inference and get cluster

Determine the maximum number of mutation combinations and write it to the **max_mutations** key in the **inference** section of ``config/config.json``. 

```bash
cd 03_inference
bash 01_generate_unpredicted_muts_csv.sh
bash 02_inference.sh
bash 03_inference_ensemble.sh

# Finally, only the one with the highest average fitness prediction value of the same site combination is retained, and then all site combinations are sorted in ascending order according to the standard deviation of the fitness value to understand the reliability of the prediction.

bash 04_get_cluster_csv.sh
cd ../
```