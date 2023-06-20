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
from caikit_template.data_model.document_rerank import DocumentRerankPrediction, SentenceRerankDocuments, SentenceRerankDocument
from caikit_template.toolkit.colbert.infra.config import ColBERTConfig
from caikit_template.toolkit.colbert.searcher import Searcher

import numpy as np
from typing import List, Dict, Union
import os

logger = alog.use_channel("<SMPL_BLK>")
error = error_handler.get(logger)

#TextQueries = Union[str, List[str], Dict[int, str], Queries]
# tokenizer = AutoTokenizer.from_pretrained("vespa-engine/colbert-medium")
# model = ColBERT.from_pretrained("vespa-engine/colbert-medium")


@task(
    required_parameters={
        "queries": List[str],
        "documents": SentenceRerankDocuments,
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
        tokenizer,
        max_num_documents=180,
        query_maxlen=32,
        include_title=False
    ) -> None:
        """Function to initialize the Reranker.
        This function gets called by `.load` and `.train` function
        which initializes this module.
        """

        super().__init__()
        self.model = model
        self._tokenzier = tokenizer
        self.max_num_documents = max_num_documents
        self.query_maxlen = query_maxlen
        self.include_title = include_title

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
            artifact = os.path.join(model_path, config.artifact)
            if not os.path.isdir(artifact):
                return cls.bootstrap(config.artifact)
            else:
                return cls.bootstrap(artifact)


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
            doc_maxlen=cls.doc_maxlen,
            query_maxlen = cls.query_maxlen
        )

        model = Searcher(
            None,
            checkpoint=pretrained_model_name_or_path,
            collection=None,
            config=config,
            rescore_only=True
        )
        return cls(model)

    # def run(self, queries: List[str], documents: List[str]) -> DocumentRerankPrediction:
    def run(self, queries: List[str], documents: SentenceRerankDocuments, *args, **kwargs) -> DocumentRerankPrediction:
        """Run inference on model.
        Args:
            queries: List[str]
            documents:  SentenceRerankDocuments
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
        for query, docs in zip(queries, documents):
            texts = []
            for p in docs:
                if include_title and 'title' in p['document'] and p['document']['title'] is not None and len(p['document']['title'].strip()) > 0:
                    texts.append(p['document']['title'] + '\n\n' + p['document']['text'])
                else:
                    texts.append(p['document']['text'])

            scores = self.model.rescore(query, texts).tolist()
            ranked_passage_indexes = np.array(scores).argsort()[::-1][:max_num_documents if max_num_documents > 0 else len(scores)].tolist()

            results = []
            for idx in ranked_passage_indexes:
                docs[idx]['score'] = scores[idx]
                results.append(docs[idx])
            ranking_results.append(results)

        return ranking_results
    
    # def encode(self, text: TextQueries):
    #     queries = text if isinstance(text, list) else [text]
    #     bsize = 128 if len(queries) > 128 else None

    #     self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
    #     Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True)

    #     return Q

    # def save(self, model_path, *args, **kwargs):
    #     """Function to save model in caikit format.
    #     This will generate store models on disk in a folder, which would be directly
    #     consumable by caikit.runtime framework.

    #     Args:
    #         model_path: str
    #             Path to store model into
    #     """
    #     module_saver = ModuleSaver(
    #         self,
    #         model_path=model_path,
    #     )
    #     with module_saver:
    #         rel_path, _ = module_saver.add_dir("watbert_model")
    #         module_saver.update_config({"watbert_artifact_path": rel_path})

    # @classmethod
    # def bootstrap(cls, pretrained_model_path):
    #     """Optional: Function that allows to load a non-caikit model artifact
    #     such as open source models from TF hub or HF and load them into
    #     this module.
    #     """
    #     # Replace following with model load code such as `transformers.from_pretrained`
    #     model = None
    #     return cls(model)
