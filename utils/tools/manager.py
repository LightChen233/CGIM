import torch
from transformers import BertModel, BertTokenizer


class PretrainedModelManager(object):
    def __init__(self, args):
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert.location)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert.location)
        if args.train.gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.pad = self.tokenizer.pad_token
        self.sep = self.tokenizer.sep_token
        self.cls = self.tokenizer.cls_token
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.special_tokens = ["[SOK]", "[EOK]", "[SOR]", "[EOR]", "[USR]", "[SYS]"]
        # SOK: start of knowledge base
        # EOK: end of knowledge base
        # SOR: start of row
        # EOR: end of row
        # USR: start of user turn
        # SYS: start of system turn

    def get_info(self, batch, info):
        infos = [item[info] for item in batch]
        last_responses = [item['last_response'] for item in batch]
        tokenized = self.tokenizer(infos, last_responses, truncation='only_first', padding=True,
                                   return_tensors='pt',
                                   max_length=self.tokenizer.max_model_input_sizes[self.args.bert.location],
                                   return_token_type_ids=True)
        tokenized = tokenized.data
        return tokenized['input_ids'].to(self.device), tokenized['token_type_ids'].to(self.device), tokenized[
            'attention_mask'].to(self.device)
