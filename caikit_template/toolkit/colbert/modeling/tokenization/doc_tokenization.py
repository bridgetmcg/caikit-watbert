import torch


from caikit_template.toolkit.colbert.modeling.hf_colbert import HF_ColBERT
from caikit_template.toolkit.colbert.infra import ColBERTConfig
from caikit_template.toolkit.colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length
from caikit_template.toolkit.colbert.utils.utils import print_message

class DocTokenizer():
    # def __init__(self, config: ColBERTConfig):
    def __init__(self, doc_maxlen, model_type):
        # self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)
        # assert False
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(model_type)

        # self.config = config
        # self.doc_maxlen = config.doc_maxlen
        self.doc_maxlen = doc_maxlen

        self.D_marker_token, self.D_marker_token_id = '[D]', self.tok.convert_tokens_to_ids('[unused1]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

        assert self.D_marker_token_id == 2
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        # assert False

        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [D] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='longest', truncation='longest_first',
                       return_tensors='pt', max_length=self.doc_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        if not self.used:
            self.used = True
            print_message("#> BERT DocTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==")
            print_message(f"#> Input: {batch_text[0]}, \t\t {bsize}")
            print_message(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print_message(f"#> Output Mask: {mask[0].size()}, {mask[0]}")

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask
