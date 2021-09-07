# DNA 4mC Site Prediction Method based on Strided Convolution
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

This project is my master theory about predicting DNA N4-methylcytosine (4mC).

## Installation
---
check out the repo,

    git clone https://github.com/Lin-YT/DNA-N4methylcytosine.git

and then install dependencies on environment (We recommended use virtualenv for development)

    pip intall -r requirement.txt

## Getting started
---
1. Known information of data:

    - The Data format is FASTA
    - There are two datasets, one for training and the other for testing
    - There are six species including _C.elegans_, _D.melanogaster_, _A.thaliana_, _E.coli_, _G.pickeringii_, _G.subterraneus_ in both datasets.
    - The Binary methylation state of the cytosine in the center of DNA (0=unmethylation, 1=methylated)

 2. Get data using `build_data.sh`, downloaded data stored in `./data`

        bash build_data.sh

3. Customize your own training config or using `config.yml` in `./config/experiments`

        python3 main.py --cfg config.yml
    Trained model files will be stored in `./results`

4. Using `test.py` to evaluate model performances.

        python3 predict.py --trained_model_name C.elegans
                           --predicted_data_name C.elegans
                           --model_path ./results
                           --train_data_path ./data/train_data
                           --test_data_path ./data/test_data
---
## Contact

- crystal20070805@gmail.com