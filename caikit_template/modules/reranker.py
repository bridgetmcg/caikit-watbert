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
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, TaskBase, module, task
from caikit.core.data_model import DataStream
from caikit.core.toolkit.errors import error_handler
from caikit_template.data_model.document_rerank import DocumentRerankPrediction, SentenceRerankPrediction, SentenceRerankDocumentsList, SentenceRerankDocuments, SentenceRerankDocument
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher

import numpy as np
from typing import List, Dict, Union
import os

logger = alog.use_channel("<SMPL_BLK>")
error = error_handler.get(logger)

@task(
    required_parameters={
        "queries": List[str],
        "documents": SentenceRerankDocumentsList,
    },
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

    def __init__(
        self, 
        model,
        doc_maxlen=180,
        query_maxlen=32,
        include_title=False, 
        max_num_documents=3
    ) -> None:
        """Function to initialize the Reranker.
        This function gets called by `.load` and `.train` function
        which initializes this module.
        """

        super().__init__()
        self.model = model
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.include_title = include_title
        self.max_num_documents = max_num_documents

    @classmethod
    def load(cls, model_path: str):
        """Load a model from disk.
        Args:
            model_path: str
                The path to the directory where the model is to be loaded from.
        Returns:
            RerankTask
        """
        if not os.path.isdir(model_path):
            return cls.bootstrap(model_path)
        else:
            config = ModuleConfig.load(model_path)
            return cls.bootstrap(model_path)


    @classmethod
    def bootstrap(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a non-caikit model
        Args:
            pretrained_model_name_or_path: str
                Path to non-caikit model.
        """

        config = ColBERTConfig(
            index_root=None,
            index_name=None,
            index_path=None,
            doc_maxlen=180,
            query_maxlen = 32
        )

        model = Searcher(
            None,
            checkpoint=pretrained_model_name_or_path + "/watbert.dnn.model",
            collection=None,
            config=config,
            rescore_only=True
        )
        return cls(model)

    # def run(self, queries: List[str], documents: List[str]) -> DocumentRerankPrediction:
    def run(self, queries: List[str], documents: SentenceRerankDocumentsList, *args, **kwargs) -> DocumentRerankPrediction:
        """Run inference on model.
        Args:
            queries: List[str]
            documents:  SentenceRerankDocumentsList
            max_num_documents: int
                Optional
            include_title: boolean 
                Optional
        Returns:
            DocumentRerankPrediction
        """

        print(f"RUN: {queries=} {documents=}")

        max_num_documents = (
            kwargs["max_num_documents"]
            if "max_num_documents" in kwargs
            else self.max_num_documents
        )

        include_title = (
            kwargs["include_title"]
            if "include_title" in kwargs
            else self.include_title
        )

        ranking_results = []
        for query, docs in zip(queries, documents.documents):
            texts = []
            for p in docs.documents:
                if include_title and 'title' in p['document'] and p['document']['title'] is not None and len(p['document']['title'].strip()) > 0:
                    texts.append(p['document']['title'] + '\n\n' + p['document']['text'])
                else:
                    texts.append(p.document.text)

            scores = self.model.rescore(query, texts).tolist()
            ranked_passage_indexes = np.array(scores).argsort()[::-1][:max_num_documents if max_num_documents > 0 else len(scores)].tolist()

            results = []
            for idx in ranked_passage_indexes:
                docs.documents[idx].score = scores[idx]
                print(type(docs.documents[idx]))
                results.append(docs.documents[idx])
            ranking_results.append(SentenceRerankPrediction(query, SentenceRerankDocumentsList([SentenceRerankDocuments(results)])))  # TODO: Is sentence=query correct here?

        return DocumentRerankPrediction(results=ranking_results)
    