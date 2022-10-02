# streamlit-demo

ref:

https://zenn.dev/littledarwin/articles/581c15b3a061cb

## Usage

1. Copy from secrets_config/sample.secrets.toml to config/secrets.toml, and input value

```toml
env='develop'
hashed_text='5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9'

[notion_service]
access_token=''
database_id=''

[aws_service]
access_key_id=''
secret_access_key=''
region_name=''

[hugging_face_service]
access_token=''

[google_custom_search_api]
cx=''
key=''
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