# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third Party
import grpc
from os import path
import sys

# Local
import caikit
from caikit.config import configure
from caikit.runtime.service_factory import ServicePackageFactory

# Since the `caikit_template` package is not installed and it is not present in path,
# we are adding it directly
sys.path.append(
    path.abspath(path.join(path.dirname(__file__), "../../"))
)

# Load configuration for Caikit runtime
CONFIG_PATH = path.realpath(
    path.join(path.dirname(__file__), "config.yml")
)
caikit.configure(CONFIG_PATH)

# NOTE: The model id needs to be a path to folder.
# NOTE: This is relative path to the models directory
MODEL_ID = "watbert"

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
)

query = "what is the color of the horse?"

documents = [{'document': {'text': 'A man is eating food.', 'title': 'A', 'docid': '0'}, 'score': 0}, 
    {'document': {'text': 'Someone in a gorilla costume is playing a set of drums.', 'title': 'in', 'docid': '1'}, 'score': 1}, 
    {'document': {'text': 'A monkey is playing drums.', 'title': 'is', 'docid': '2'}, 'score': 2}, 
    {'document': {'text': 'A man is riding a white horse on an enclosed ground.', 'title': 'riding', 'docid': '3'}, 'score': 3}, 
    {'document': {'text': 'Two men pushed carts through the woods.', 'title': 'through', 'docid': '4'}, 'score': 4}]


port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")
client_stub = inference_service.stub_class(channel)

## Create request object
request = inference_service.messages.RerankTaskRequest(queries=query, documents=documents)

## Fetch predictions from server (infer)
response = client_stub.RerankTaskPredict(
    request, metadata=[("mm-model-id", MODEL_ID)]
)

## Print response
print("RESPONSE:", response)