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

from caikit.core import (
    dataobject,
    DataObjectBase,
)

from typing import List, Dict


documents = [{'document': {'text': 'A man is eating food.', 'title': 'A', 'docid': '0'}, 'score': 0},
             {'document': {'text': 'Someone in a gorilla costume is playing a set of drums.', 'title': 'in', 'docid': '1'}, 'score': 1},
             {'document': {'text': 'A monkey is playing drums.', 'title': 'is', 'docid': '2'}, 'score': 2},
             {'document': {'text': 'A man is riding a white horse on an enclosed ground.', 'title': 'riding', 'docid': '3'}, 'score': 3},
             {'document': {'text': 'Two men pushed carts through the woods.', 'title': 'through', 'docid': '4'}, 'score': 4}]

# @dataobject()
# class SentenceRerankDict(DataObjectBase):
#     """An input document"""

#     text: str
#     title: str
#     docid: str

@dataobject()
class SentenceRerankDocument(DataObjectBase):
    """An input document"""

    document: Dict[str,str]
    score: int

@dataobject()
class SentenceRerankDocuments(DataObjectBase):
    """The input documents"""

    documents: List[SentenceRerankDocument]

@dataobject()
class SentenceRerankDocumentsList(DataObjectBase):
    """The input documents"""

    documents: List[SentenceRerankDocuments]

@dataobject()
class SentenceRerankPrediction(DataObjectBase):
    """The result of a similarity scores prediction."""

    sentence: str
    result: List[str]

@dataobject()
class DocumentRerankPrediction(DataObjectBase):
    """The result of a similarity scores prediction."""

    results: List[SentenceRerankPrediction]
