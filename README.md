# distributed-recsys
Distributed Recommender System

Prerequisite - Distributed GPU cluster setup in GCP

## Setup
```
git clone https://github.com/funktor/distributed-recsys.git
cd distributed-recsys
pip install -r requirements.txt
pip install --upgrade Cython
python setup_ml_32m_gcp.py bdist_wheel
pip install --force-reinstall dist/*.whl
```

## Download ML-32M Datasets
```
cd /tmp
wget https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip
mkdir ~/distributed-recsys/datasets
mv ml-32m ~/distributed-recsys/datasets
```

## Run Data Generation Pipeline
```
cd ~/distributed-recsys
python data_generator.py \
    --dataset_path_local "dataset_ml_32m" \
    --gcs_bucket "recsys" \
    --gcs_prefix "ml-32m" \
    --gcs_data_dir "dataset_ml_32m" \
    --min_num_ratings 20
```

## Run on a single node with 8 GPUs
```
cd ~/distributed-recsys
nohup torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=8 \
        trainer.py \
            --gcs_bucket "recsys" \
            --gcs_prefix "ml-32m"  \
            --gcs_data_dir "dataset_ml_32m" \
            --batch_size 128 \
            --num_epochs 10 \
            --num_workers 4 \
            --accumulate_grad_batches 4 \
            --model_out_dir "/tmp/model_outputs" >output.log 2>&1 &
```

## Run on 2 nodes each with 8 GPUs
```
# Get IP addresses of nodes/pods. If using kubernetes pods, run kubectl get pods -o wide
# On master node with IP address 240.76.44.135

cd ~/distributed-recsys
nohup torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=240.76.44.135 \
    --master_port=29500 \
    --nproc_per_node=8 \
    trainer.py \
        --gcs_bucket "recsys" \
        --gcs_prefix "ml-32m"  \
        --gcs_data_dir "dataset_ml_32m" \
        --batch_size 128 \
        --num_epochs 10 \
        --num_workers 4 \
        --accumulate_grad_batches 4 \
        --model_out_dir "/tmp/model_outputs" >output.log 2>&1 &


# On worker node with a different IP address. Use the same master node IP addr from above in master_addr.
nohup torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=240.76.44.135 \
    --master_port=29500 \
    --nproc_per_node=8 \
    trainer.py \
        --gcs_bucket "recsys" \
        --gcs_prefix "ml-32m"  \
        --gcs_data_dir "dataset_ml_32m" \
        --batch_size 128 \
        --num_epochs 10 \
        --num_workers 4 \
        --accumulate_grad_batches 4 \
        --model_out_dir "/tmp/model_outputs" >output.log 2>&1 &
```

Using torchrun, we need to individually log into the nodes/pods and run the commands. Another option that does not require to log in to each node is to use the `mpirun` command. Here I have a head node and 2 worker nodes. I run the commands only from head node.

## Setup MPI (on head node as well as worker nodes)
```
sudo apt-get update
sudo apt-get install openmpi-bin libopenmpi-dev
mpirun --version
```

## Setup SSH and TCP on head node and worker nodes
```
# On each worker node start ssh server
sudo apt update
sudo apt install openssh-server -y
sudo service ssh start

# Generate SSH key on head node
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Copy public key from head node to worker nodes
# On each worker node
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Append the copied key (paste the output from 'cat ~/.ssh/id_rsa.pub' in head node)
# On each worker node
echo "<head node public key>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## Run on head node
```
# Each worker node has 8 GPUs total 16 GPUs across 2 nodes
nohup mpirun -np 16 \
    -H 240.76.37.135:8, 240.76.41.135:8 \
    -x MASTER_ADDR=240.76.37.135 \
    -x MASTER_PORT=29500 \
    -x PATH \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    python \
        trainer.py \
            --gcs_bucket "recsys" \
            --gcs_prefix "ml-32m"  \
            --gcs_data_dir "dataset_ml_32m" \
            --batch_size 128 \
            --num_epochs 10 \
            --num_workers 4 \
            --accumulate_grad_batches 4 \
            --model_out_dir "/tmp/model_outputs" >output.log 2>&1 &
```