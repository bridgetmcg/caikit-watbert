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

Step 5: Add model to the `demo\models\watbert` folder. The model is expected to be named `watbert.dnn.model` in this demo.


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

RESPONSE: results {
  sentence: "what is the color of the horse?"
  result {
    documents {
      documents {
        document {
          text: "A man is riding a white horse on an enclosed ground."
          title: "riding"
          docid: "3"
        }
        score: 18.641214370727539
      }
      documents {
        document {
          text: "Someone in a gorilla costume is playing a set of drums."
          title: "in"
          docid: "1"
        }
        score: 10.520210266113281
      }
      documents {
        document {
          text: "A monkey is playing drums."
          title: "is"
          docid: "2"
        }
        score: 9.5272674560546875
      }
    }
  }
}
```