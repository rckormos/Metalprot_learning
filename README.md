# Metal Binding Classifier and Regressor
NN to classify metal binding sites. DeGrado Lab 2021.

## Installation
This repo is pip installable. To re-train the model, you will need to install this module as well as the requirements listed in <code>requirements.txt</code>. Here, I'll walk through how you should go about doing this.

1) Clone Repo if you haven't done so already. Use the following command: <code>git clone [link to Metalprot_learning]</code>

2) In the root directory of the repo, install requirements using the following command: <code>pip install -r requirements.txt</code>. I recommend creating a new virtual environment for this. This may take a few minutes.

3) To install the repo, run the following command: <code>pip install .</code> in the root directory of the repo.

## Running Jobs
The job scripts for input featurization, processing, and training are located in <code>Metalprot_learning/job_scripts</code>. 