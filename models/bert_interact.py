import logging
import pprint
import random
import fitlog
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AdamW
from models.interactive_block import CycleGuidedInteractiveLearningModule
import utils.tools.tool
from utils.tools.manager import PretrainedModelManager
from utils.tools.recorder import Recoder
from utils.tools.tool import Batch
from tqdm import tqdm
import re
import os
from utils.evaluate import EvaluateTool
from utils.loader import DatasetTool


class CGIM(torch.nn.Module):
    def __init__(self, args, inputs):
        super().__init__()
        self.args = args
        self.DatasetTool = DatasetTool
        self.EvaluateTool = EvaluateTool
        _, _, _, entities = inputs
        # Pretrained Model Preparation
        manager = PretrainedModelManager(self.args)
        self.bert = manager.bert
        self.tokenizer = manager.tokenizer
        special_tokens_dict = {'additional_special_tokens': manager.special_tokens + entities}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.get_info = manager.get_info
        # Model Definition
        self.hidden_size = args.train.hidden_size
        self.pretrain_size = args.train.pretrain_size
        self.models = nn.ModuleList(
            [CycleGuidedInteractiveLearningModule(args).to(self.device) for _ in range(args.train.layer_num)])
        self.w_hi = nn.Linear(self.hidden_size * 3, 2)
        self.w_qi = nn.Linear(self.hidden_size * 3, 2)
        self.w_kbi = nn.Linear(self.hidden_size * 3, 2)
        self.w_h = nn.Linear(self.pretrain_size, self.hidden_size)
        self.w_q = nn.Linear(self.pretrain_size, self.hidden_size)
        self.w_kb = nn.Linear(self.pretrain_size, self.hidden_size)
        self.criterion = nn.BCELoss()
        self.recoder = Recoder("out/", self.args, 1)

    @property
    def device(self):
        if self.args.train.gpu:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        all_params = set(self.parameters())
        params = [{"params": list(all_params), "lr": self.args.lr.bert}]
        self.optimizer = AdamW(params)

    def forward(self, batch):
        # Get KB Encoded Vector
        kb_token_ids, kb_type_ids, kb_mask_ids = self.get_info(batch, 'knowledge_base')
        h_kb, utt_kb = self.bert(input_ids=kb_token_ids, token_type_ids=kb_type_ids, attention_mask=kb_mask_ids)
        # Get Query Encoded Vector
        q_token_ids, q_type_ids, q_mask_ids = self.get_info(batch, 'query')
        h_q, utt_q = self.bert(input_ids=q_token_ids, token_type_ids=q_type_ids, attention_mask=q_mask_ids)
        # Get History Encoded Vector
        h_token_ids, h_type_ids, h_mask_ids = self.get_info(batch, 'history')
        h_h, utt_h = self.bert(input_ids=h_token_ids, token_type_ids=h_type_ids, attention_mask=h_mask_ids)

        # Get Hidden Interactive Encoded Vectors
        hidden = torch.cat((utt_q.view(-1, 1, self.pretrain_size),
                            utt_h.view(-1, 1, self.pretrain_size),
                            utt_kb.view(-1, 1, self.pretrain_size)), dim=1)
        hidden_q = self.w_q(hidden)
        hidden_h = self.w_h(hidden)
        hidden_kb = self.w_kb(hidden)

        # Encoded by circle transformer layers
        for model in self.models:
            hidden_q, hidden_kb, hidden_h = model(hidden_state_q=hidden_q, hidden_state_h=hidden_h,
                                                  hidden_state_kb=hidden_kb)
        # Linear decode layer
        out_qi = self.w_qi(hidden_q.view(-1, self.hidden_size * 3))
        out_hi = self.w_hi(hidden_h.view(-1, self.hidden_size * 3))
        out_kbi = self.w_kbi(hidden_kb.view(-1, self.hidden_size * 3))
        out = []
        for qi, hi, kbi in zip(out_qi, out_hi, out_kbi):
            out.append([qi.argmax().data.tolist(), hi.argmax().data.tolist(), kbi.argmax().data.tolist()])
        loss = torch.Tensor([0]).to(self.device)
        if self.training:
            loss = F.cross_entropy(out_qi,
                                   torch.Tensor(
                                       utils.tools.tool.in_each(batch, lambda x: x["consistency"][0])).long().to(
                                       self.device)) \
                   + F.cross_entropy(out_hi,
                                     torch.Tensor(
                                         utils.tools.tool.in_each(batch, lambda x: x["consistency"][1])).long().to(
                                         self.device)) \
                   + F.cross_entropy(out_kbi,
                                     torch.Tensor(
                                         utils.tools.tool.in_each(batch, lambda x: x["consistency"][2])).long().to(
                                         self.device))
        return loss, out

    def start(self, inputs):
        train, dev, test, _ = inputs
        if (self.args.model.resume is not None) and (self.args.model.resume != 'None'):
            self.load(self.args.model.resume)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        self.run_eval(train, dev, test)

    def run_test(self, dataset):
        self.eval()
        all_out = []
        all_size = 0
        all_cos = {}
        if self.args.train.compute_cos:
            all_cos["cos_q_kb"] = 0
            all_cos["cos_h_kb"] = 0
            all_cos["cos_q_h"] = 0

        for batch in tqdm(Batch.to_list(dataset, self.args.train.batch)[0: len(dataset)]):
            loss, out = self.forward(batch)
            all_out += self.get_pred(out)
            all_size += len(batch)
        return self.EvaluateTool.evaluate(all_out, dataset, self.args), all_out

    def run_eval(self, train, dev, test):
        logging.info("Starting evaluation")
        self.eval()
        summary = {}
        ds = {"test": test}
        for set_name, dataset in ds.items():
            tmp_summary, pred = self.run_test(dataset)
            self.DatasetTool.record(pred, dataset, set_name, self.args)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
        logging.info(pprint.pformat(summary))

    def run_batches(self, dataset, epoch):
        all_loss = 0
        all_size = 0
        iteration = 0
        all_cos = {}
        if self.args.train.compute_cos:
            all_cos["cos_q_kb"] = 0
            all_cos["cos_h_kb"] = 0
            all_cos["cos_q_h"] = 0

        for batch in tqdm(Batch.to_list(dataset, self.args.train.batch)[0: len(dataset)]):
            loss, _, = self.forward(batch)
            self.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_loss += loss.item()
            iteration += 1
            all_size += len(batch)
        return all_loss / all_size, iteration

    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        count = 0
        for param in self.bert.parameters():
            param.requires_grad = True
        epochs = self.args.train.epoch
        for epoch in range(epochs):
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = {"epoch": epoch, "iteration": iteration}
            loss, iter = self.run_batches(train, epoch)
            fitlog.add_loss({"train_loss": loss}, step=epoch)
            summary.update({"loss": loss})
            iteration += iter

            set_name = "dev"
            dataset = dev
            tmp_summary, pred = self.run_test(dataset)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
            fitlog.add_metric({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()}, step=epoch)
            data = self.recoder.get_error_pred(dataset, pred)
            if self.recoder.record([summary['eval_dev_overall_acc']], state_dict=self.state_dict(), data=data):
                count = 0
                self.recoder.print_output()
                set_name = "test"
                dataset = test
                tmp_summary, pred = self.run_test(dataset)
                summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
                fitlog.add_best_metric({"QI F1": summary['eval_test_f1_qi']})
                fitlog.add_best_metric({"HI F1": summary['eval_test_f1_hi']})
                fitlog.add_best_metric({"KBI F1": summary['eval_test_f1_kbi']})
                fitlog.add_best_metric({"Overall": summary['eval_test_overall_acc']})
            else:
                count += 1
                if count == self.args.train.early_stop:
                    return

    def load(self, file):
        logging.info("Loading models from {}".format(file))
        state = torch.load(file)
        model_state = state["models"]
        self.load_state_dict(model_state)

    def get_pred(self, out):
        pred = []
        for ele in out:
            pred.append(ele)
        return pred
