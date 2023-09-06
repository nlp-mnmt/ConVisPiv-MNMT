# ConVisPiv-MNMT
Multi-grained Visual Pivot-guided Multi-modal Neural Machine Translation with Text-aware Cross-modal Contrastive Disentangling
# Requirements
cuda==11.2  
python==3.7  
torch==1.8.1
# dataset
txt data we employ the data set [Multi30K dataset](http://www.statmt.org/wmt18/multimodal-task.html), then use [BPE](https://github.com/rsennrich/subword-nmt) to preprocess the raw data(dataset/data/task1/tok/). Image features are extracted through the pre-trained Resnet-101.  
The data-raw folder above is the data processed by BPE.
##### BPE (learn_joint_bpe_and_vocab.py and apply_bpe.py)
English, German, French use BPE participle separately.   
# ConVisPiv-MNMT Quickstart
Step 1: bash data-preprocess.sh Then add the pre-trained Resnet-101 image feature to $DATA_DIR

step 2: bash data-train.sh

step 3: bash data-checkpoints.sh

step 4: bash data-generate.sh

The data-bin folder is the text data processed by bash data-preprocess.sh. Add the extracted image features here to start training the model.
