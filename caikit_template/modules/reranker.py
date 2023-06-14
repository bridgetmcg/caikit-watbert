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

import alog
from caikit.core import ModuleBase, ModuleLoader, ModuleSaver, TaskBase, module, task
from caikit.core.data_model import DataStream
from caikit.core.toolkit.errors import error_handler
from caikit_template.data_model.document_rerank import (DocumentRerankPrediction)
from primeqa.components.reranker.colbert_reranker import ColBERTReranker
from typing import List, Dict
import numpy as np
import os

logger = alog.use_channel("<SMPL_BLK>")
error = error_handler.get(logger)

@task(
    required_parameters={"queries": List[str],
                    "documents":  List[List[Dict]]},
    output_type=DocumentRerankPrediction,
)
class RerankTask(TaskBase):
    pass

@module(
    "00110203-0405-0607-0809-0a0b02dd0e0f",
    "RerankerModule",
    "0.0.1",
    RerankTask,
)
class Rerank(ModuleBase):

    def __init__(self, model=None) -> None:
        """Function to initialize the Reranker.
        This function gets called by `.load` and `.train` function
        which initializes this module.
        """

        super().__init__()
        self.model = model

    @classmethod
    def load(cls, model_path: str, **kwargs):
        """Load a caikit model
        Args:
            model_path: str
                Path to caikit model.
        """
        reranker = ColBERTReranker(model=model_path)
        model = reranker.load()
        return cls(model)

    def run(self, queries: List[str],
                    documents:  List[List[Dict]], *args, **kwargs) -> DocumentRerankPrediction:
        """Run inference on model.
        Args:
            queries: List[str],
            documents:  List[List[Dict]]
        Returns:
            DocumentRerankPrediction
        """
        # This is the main function used for inferencing.
        return self.model.predict([queries], documents=[documents],max_num_documents=2)

    def save(self, model_path, *args, **kwargs):
        """Function to save model in caikit format.
        This will generate store models on disk in a folder, which would be directly
        consumable by caikit.runtime framework.

        Args:
            model_path: str
                Path to store model into
        """
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            rel_path, _ = module_saver.add_dir("watbert_model")
            module_saver.update_config({"watbert_artifact_path": rel_path})

    @classmethod
    def bootstrap(cls, pretrained_model_path):
        """Optional: Function that allows to load a non-caikit model artifact
        such as open source models from TF hub or HF and load them into
        this module.
        """
        # Replace following with model load code such as `transformers.from_pretrained`
        model = None
        return cls(model)
