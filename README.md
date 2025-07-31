# Youtube-Sentimental-Insights


## AWS-CICD-Deployment-with-Github-Actions
### 1. Login to AWS console.

### 2. Create IAM user for deployment

#### with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws
#### Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

#### Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

### 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/youtube

### 4. Create EC2 machine (Ubuntu) 

### 5. Open EC2 and Install docker in EC2 Machine:
#### optinal
    ```bash
	sudo apt-get update -y
    ```
    ```bash
	sudo apt-get upgrade
    ```
#### required
    ```bash
	curl -fsSL https://get.docker.com -o get-docker.sh
    ```
    ```bash
	sudo sh get-docker.sh
    ```
    ```bash
	sudo usermod -aG docker ubuntu
    ```
    ```bash
	newgrp docker
    ```
	
### 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


### 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app



## AWS 
```bash
aws configure
```

## DVC STEPS:
```bash
dvc init
dvc repro
dvc dag
```

## Initial STEPS:
Clone the repository 
```bash
https://github.com/Bavan-M/Youtube-Sentimental-Insights
```
### STEP-01 Create a Virtual environment after opening the repository

```bash
python -m venv youtubeInsights 
```

```bash
.\youtubeInsights\Scripts\activate
```
### STEP-02 Install the Requirments
```bash
pip install -r requirements.txt
