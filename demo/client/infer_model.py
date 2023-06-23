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

from os import path
import sys

# Third Party
import grpc
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message_factory import MessageFactory
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)

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
configure(CONFIG_PATH)

# NOTE: The model id needs to be a path to folder.
# NOTE: This is relative path to the models directory
MODEL_ID = "wat"

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
)

queries = ["what is the color of the horse?"]

documents = [
    {'document': {'text': 'A man is eating food.', 'title': 'A', 'docid': '0'}, 'score': 0},
    {'document': {'text': 'Someone in a gorilla costume is playing a set of drums.', 'title': 'in', 'docid': '1'}, 'score': 1},
    {'document': {'text': 'A monkey is playing drums.', 'title': 'is', 'docid': '2'}, 'score': 2},
    {'document': {'text': 'A man is riding a white horse on an enclosed ground.', 'title': 'riding', 'docid': '3'}, 'score': 3}, 
    {'document': {'text': 'Two men pushed carts through the woods.', 'title': 'through', 'docid': '4'}, 'score': 4},
]


port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")
client_stub = inference_service.stub_class(channel)
reflection_db = ProtoReflectionDescriptorDatabase(channel)
desc_pool = DescriptorPool(reflection_db)
services = [
    x for x in reflection_db.get_services() if x.startswith("caikit.runtime.") and not x.endswith("TrainingService") and not x.endswith("TrainingManagement")
]
if len(services) != 1:
    print(f"Error: Expected 1 caikit.runtime service, but found {len(services)}.")
service_name = services[0]
service_prefix, _, _ = service_name.rpartition(".")

## Create request object
request_name = f"{service_prefix}.RerankTaskRequest"

request_desc = desc_pool.FindMessageTypeByName(request_name)
srds_desc = desc_pool.FindMessageTypeByName("caikit_data_model.SentenceRerankDocuments")
srd_desc = desc_pool.FindMessageTypeByName("caikit_data_model.SentenceRerankDocument")
srdl_desc = desc_pool.FindMessageTypeByName("caikit_data_model.SentenceRerankDocumentsList")
request = MessageFactory(desc_pool).GetPrototype(request_desc)
sentenceRerankDocuments = MessageFactory(desc_pool).GetPrototype(srds_desc)
sentenceRerankDocument = MessageFactory(desc_pool).GetPrototype(srd_desc)
sentenceRerankDocumentsList = MessageFactory(desc_pool).GetPrototype(srdl_desc)

srds = sentenceRerankDocuments(documents=[sentenceRerankDocument(document=x["document"], score=x["score"]) for x in documents])
srdl = sentenceRerankDocumentsList(documents=[srds])
## Fetch predictions from server (infer)
response = client_stub.RerankTaskPredict(request(queries=queries, documents=srdl), metadata=[("mm-model-id", MODEL_ID)])

## Print response
print("RESPONSE:", response)
