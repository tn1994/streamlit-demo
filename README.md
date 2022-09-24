# streamlit-demo

ref:

https://zenn.dev/littledarwin/articles/581c15b3a061cb

## Usage

1. Copy from secrets_config/sample.secrets.toml to config/secrets.toml, and input value

```toml
# set up streamlit
env='develop'
hashed_text='5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9'

# Notion Service
notion_access_token='...'
notion_database_id='...'

# AWS Service
aws_access_key_id='...'
aws_secret_access_key='...'
region_name='...'

# Hugging Face Service
hugging_face_access_token='...'
```

2. Run build.sh

```shell
sh build.sh
```

3. Run compose_up.sh

```shell
sh compose_up.sh
```

4. Access to localhost:8501

[Access](http://localhost:8501/)

## About requirements.txt

```shell
# Basic
streamlit
pandas
numpy
matplotlib
scikit-learn
seaborn

# Stock Service
jpholiday
pandas_datareader

# AWS Service
boto3

# Hugging Face Service
transformers
sentencepiece
torch
torchvision

# KeyBERT Service
keybert
sentence-transformers
mecab-python3
unidic-lite
```