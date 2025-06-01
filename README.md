## Replication Package
This is the open-source repository of the paper titled **Demystifying Checker Bugs in Deep Learning Libraries** Submitted
to the **TOSEM** journal. 

:wave: We are also constantly updating the repository to improve the package's reproducibility for other researchers.

[//]: # (<img src="https://github.com/icse2026/blob/master/assets/ML_APR_flow_new.png" width="600" class="center">)

### Data
You can access all of our data as follows:
#### Prompt templates

[Directed Chain of Thought- D-CoT](https://github.com/dmc1778/icse2026/blob/master/assets/D-CoT.txt)

[Directed Augmented Chain of Thought - DA-CoT](https://github.com/dmc1778/icse2026/blob/master/assets/DA-CoT.txt)

[Directed Fewshot Chain of Thought - DF-CoT](https://github.com/dmc1778/icse2026/blob/master/assets/DF-CoT.txt)

#### Keywords for filtering checker bugs
[Keywords](https://github.com/dmc1778/icse2026/blob/master/assets/all_keywords.csv)<br>

#### Taxonomy of Checker Bugs
[Taxonomy data](https://github.com/dmc1778/icse2026/blob/master/assets/taxonomyData.csv)<br>

| Library | Commits used for Taxonomy Creation | Evaluation Data |
|----------|----------|----------|
| PyTorch | [PyTorch commits](https://github.com/dmc1778/icse2026/blob/master/mining/commits/pytorch/pytorch.csv) | [PyTorch test data](https://github.com/dmc1778/icse2026/blob/master/data/taxonomy_data/pytorch_test_data.json) |
| TensorFlow | [TensorFlow commits](https://github.com/dmc1778/icse2026/blob/master/mining/commits/tensorflow/tensorflow.csv) | [TensorFlow test data](https://github.com/dmc1778/icse2026/blob/master/data/taxonomy_data/tensorflow_test_data.json) |

### :hammer: Setup & Running LLM models (section 3.2 in the manuscript)
First, you need to install Python virtual environments from the following link: [venv](https://docs.python.org/3/library/venv.html).
Then clone the repository and cd to the root folder and run the following command:
```
python3 -m venv <ENV_NAME>
```
:bell: We recommend using Python 3.9 as the default version of your Python interpreter.
Then you need to activate the environment:
```
source <ENV_NAME>/bin/activate
```
Then you have to install the required dependencies:
```
pip install -r requirements.txt
```
Once all required dependencies are installed, create a ```.env``` file in the root of the project and put your API keys (OpenAI and Fireworks keys):
```
OPENAI_API_KEY=""
FIREWORKS_API_KEY=""
```
#### :rocket: Building evaluation dataset

cd to ```core``` directory. Run the following command to construct a dataset for the PyTorch library:

```
python core/build_commit_database.py pytorch pytorch taxonomy
```

And the following command for TensorFlow:

```
python core/build_commit_database.py tensorflow tensorflow taxonomy
```
#### :rocket: Running LLM models
Once you have created the datasets, run the following command to run ```gpt-4o-mini``` on the PyTorch dataset:
```
python core/run_exp_paralell.py pytorch 2 openai regular
```
To run ```deepseek-r1```, ```qwen2p5-72b-instruct```, ```qwen2p5-coder-32b-instruct```, and ```llama-v3p1-405b-instruct```:
```
python core/run_exp_paralell.py tensorflow 2 fireworks regular
```
The results will be stored within the ```output``` directory.

#### :rocket: Evaluation
To generate Figure 2a, run the following command:
```
python utils/evaluation.py pytorch fig2
```
To generate Figure 2b, run the following command:
```
python utils/evaluation.py tensorflow fig2
```
You can download the figure 2 data [here](https://github.com/dmc1778/icse2026/blob/master/output/all_results_filtered.csv)

To generate Figure 3, run the following command:
```
python utils/evaluation.py fig3
```
You can download the figure 3 data [here](https://github.com/dmc1778/icse2026/blob/master/assets/rootCauseAcc.csv)

### :hammer: Setup & Running DL Fuzz Testing Tools (section 3.3 in the manuscript)

## :no_entry: Disclaimer :no_entry:
The fuzzers used in this project generate test cases designed to identify weaknesses in DL libraries. 
We strongly recommend running these tools in a sandbox or isolated environment, as they may crash or hang your operating system. 
Please avoid using any environment containing sensitive or critical information.

## Subject Library Versions
In this project, we use the latest releases of PyTorch and TensorFlow at the time we started this project.
We use versions 2.0.0, 2.0.1, and 2.1.0 for PyTorch and versions 2.11.0, 2.12.0, 2.13.0, and 2.14.0 for TensorFlow.

## Subject APIs
We use the following DL APIs for test case generation:

[PyTorch APIs](https://github.com/dmc1778/icse2026/blob/master/data/torch_apis.txt)

[TensorFlow APIs](https://github.com/dmc1778/icse2026/blob/master/data/tf_apis.txt)

:warning: Please note that we do not run the fuzzers on all DL APIs; we only evaluate them on the specific APIs mentioned.
These APIs are known to cause bugs in the subject releases of PyTorch and TensorFlow. 
If you wish to run the fuzzers on all APIs, you can still do so. However, 
the versions of the DL fuzzers provided in this replication package have been modified to focus exclusively on the mentioned APIs.

## DL Fuzzers
In this project, we target two groups of DL fuzzers.
### Traditional DL fuzzers
The traditional DL fuzzers used in this paper are as follows:

[FreeFuzz](https://github.com/ise-uiuc/FreeFuzz)

[DeepRel](https://github.com/ise-uiuc/DeepREL)

[ACETest](https://github.com/shijy16/ACETest)

### LLM-based DL fuzzers
We also used two recently introduced LLM-based DL fuzzers:

[TitanFuzz](https://github.com/ise-uiuc/TitanFuzz)

[FuzzGPT](https://figshare.com/s/fc28098a692f24fb4b39)


## Getting started
### Create conda environments and install the required packages
To run the fuzzers, you need multiple conda environments. So, please install Anaconda3 for Linux from the [this](https://docs.anaconda.com/anaconda/install/linux/) link.

After you have successfully installed Anaconda3 for Linux, run the following commands to create environments and install the required packages:

To create the environments, run the following commands:
```
bash create_envs_torch.sh
bash create_envs_tf.sh
```
:warning: Please note that all of the required packages for all of the subject fuzzers are installed once you create the conda environment.

Before running the fuzzers, you need to patch a bug within the astuneparse library, which is required by TitanFuzz and AtlasFuzz. Please read TitanFuzz's readme to fix the bug.

### Running DL Fuzzers and generating test cases
To run the subject DL fuzzers, you can use their source code from their replication packages. 
However, we recommend downloading the versions provided in this replication package. 
We have modified the argument parsing components to store the generated test cases for each library and its corresponding releases.
By using the versions in this package, you will save a lot of time.

You can find the versions under the ```fuzzers``` directory.

#### Running NablaFuzz
Before running FreeFuzz and DeepRel, make sure that you have installed [mongodb](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/#std-label-install-mdb-community-ubuntu) community edition for Linux. DeepRel and FreeFuzz depend on the MongoDB database as their test inputs are stored within the database. 

We use test inputs of FreeFuzz for DeepRel because these tools are from the same authors and cover the highest number of APIs for both PyTorch and TensorFlow.

Once you have installed MongoDB, run the following commands:

```
mongorestore dump/
```
Now, you have loaded the test inputs into the MongoDB database and are ready to run DeepRel and FreeFuzz.

#### Running DeepRel
To run DeepRel, please unzip the source to your desired directory and run the following command to generate test cases for PyTorch:

```
cd DeepREL/pytorch/src
python DeepREL.py 2.0.0 torch
```
For TensorFlow:
```
cd DeepREL/tensorflow/src
python DeepREL.py 2.11.0 tf
```
#### Running FreeFuzz

To run FreeFuzz on PyTorch, please run the following commands:
```
cd /FreeFuzz/src
```
Then run:
```
python FreeFuzz.py --conf=FreeFuzz/src/config/expr.conf --release=2.0.0 --library=torch --iteration_round=5
```
For TensorFlow:
```
python FreeFuzz.py --conf=FreeFuzz/src/config/expr.conf --release=2.11.0 --library=tf --iteration_round=5
```
#### Running ACETest
Please unzip the source to your desired directory and cd to the root.
To run ACETest on PyTorch, run the following command:
```
python main.py --test_round=1000 --mode=all --release=2.0.0 --framework=torch --work_path=output --filter=all
python main.py --test_round=1000 --mode=all --release=2.0.1 --framework=torch --work_path=output --filter=all
python main.py --test_round=1000 --mode=all --release=2.1.0 --framework=torch --work_path=output --filter=all
```
For TensorFlow:
```
python main.py --test_round=1000 --mode=all --release=2.11.0 --framework=tf --work_path=output --filter=all
python main.py --test_round=1000 --mode=all --release=2.12.0 --framework=tf --work_path=output --filter=all
python main.py --test_round=1000 --mode=all --release=2.13.0 --framework=tf --work_path=output --filter=all
python main.py --test_round=1000 --mode=all --release=2.14.0 --framework=tf --work_path=output --filter=all
```
#### Running TitanFuzz
First, change your directory to TitanFuzz root, then run it using the following command for the PyTorch library:
```
bash scripts/local_run.sh torch data/torch_applicable_apis.txt torch_2.0.1 and 2.0.1 5
```
For TensorFlow:
```
bash scripts/local_run.sh tf data/tf_applicable_apis.txt tf_2.11.0 and 2.11.1 5
```
#### Running FuzzGPT
