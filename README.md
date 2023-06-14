# Caikit WatBERT

Caikit runtime repository which serves a WatBERT AI model using [caikit](https://github.com/caikit/caikit).

## Run locally

The following tools are required:

- [git](https://git-scm.com/)
- [python](https://www.python.org) (v3.8+)
- [pip](https://pypi.org/project/pip/) (v23.0+)

**Note:** Before installing dependencies and to avoid conflicts in your environment, it is advisable to use a virtual environment. The subsection which follows provides an example of a virtual environment, python venv.

Step 1: Clone the project and navigate to the project folder

```shell
git clone https://github.com/bridgetmcg/caikit-watbert.git
cd caikit-watbert
```

Step 2: Set a virtual environment (optional)

```shell
python -m venv venv
```

Step 3: Activate the virtual environment

```shell
source venv/bin/activate
```

Step 4: Install the needed modules and libraries

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Starting the Caikit Runtime

In one terminal, start the runtime server:

```shell
cd demo/server
python start_runtime.py
```

You should see output similar to the following:

```ShellSession
$ python start_runtime.py

[ ... ]
{"channel": "MODEL-LOADER", "exception": null, "level": "info", "log_code": "<RUN89711114I>", "message": "Loading model 'watbert'", "num_indent": 0, "thread_id": 140704708310592, "timestamp": "2023-06-14T15:50:55.104085"}
[ ... ]
{"channel": "GRPC-SERVR", "exception": null, "level": "info", "log_code": "<RUN10001001I>", "message": "Caikit Runtime is serving on port: 8085 with thread pool size: 5", "num_indent": 0, "thread_id": 140704708310592, "timestamp": "2023-06-14T15:50:55.222336"}

```

## Inferencing with the served model

In another terminal, run the client code to infer the model:

```shell
source venv/bin/activate
cd demo/client
python infer_model.py
```

The client code calls the model and queries for generated text using text passed from the client.

You should see output similar to the following after the word `World` is passed:

```ShellSession
$ python infer_model.py

RESPONSE: greeting: "Hello World"
```

## Repository Layout

```text
├── caikit-template/:                       top-level package directory (will change to your repo name after template is deployed)
│   │── caikit_template/:                   a directory that defines Caikit module(s) that can include algorithm(s) implementation that can train/run an AI model 
│   │   ├── config/:                        a directory that contains the configuration for the module and model input and output
│   │   │   ├── config.yml:                 configuration for the module and model input and output
│   │   ├── data_model/:                    a directory that contains the data format of the Caikit module
│   │   │   ├── hello_world.py:             data class that represents the AI model attributes in code
│   │   │   ├── __init__.py:                makes the hello_world class visible in the project
│   │   ├── modules/:                       a directory that contains the Caikit module of the model
│   │   │   ├── hello_world.py:             a class that bootstraps the AI model in Caikit so it can be served and used (infer/train)
│   │   │   ├── __init__.py:                makes the hello_world class visible in the project
|   |   |── __init__.py:                    makes the data_model and runtime_model packages visible
│   │── demo/:                              a directory which contains code and configuration to test the model
│   │   │── client/:                        a directory which contains artifacts to use (infer and train) the AI model spceified in the `caikit_template` package
|   │   │   ├── config.yml:                 caikit runtime configuration file
│   │   │   ├── infer_model.py:             sample client which calls the Caikit runtime to perform inference on a model it is serving
│   │   │   ├── train_model.py:             sample client which calls the Caikit runtime to perform training on a model it is serving
│   │   │── models/:                        a directory that contains the Caikit metadata of the models and any artifacts required to run the models (usually generated after saving and should not be modified)
│   │   │   ├── hello_world/config.yml:     a metadata that defines the example Caikit model
│   │   │── server/:                        a directory which contains artifacts to start Caikit runtime
|   │   │   ├── config.yml:                 configuration for handling the model by the Caikit runtime
│   │   │   ├── start_runtime.py:           a wrapper to start the Caikit runtime as a gRPC server. The runtime will load the model at startup
|   │   ├── train_data/:                    a directory which contains the training data
|   │   |   ├── sample_data.csv:            sample training dataset to perform training of the model
└── └── requirements.txt:                   specifies library dependencies
```

