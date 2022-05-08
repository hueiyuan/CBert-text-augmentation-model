import torch
from dataclasses import dataclass, field
from transformers import BertConfig, BertForMaskedLM

@dataclass(eq=False)
class CbertModel:
    hyperparameters: dict = field(
        default=None, metadata={"help": "model hyperparameters with dict type"}
    )
    num_labels: int = field(
        default=None, metadata={"help": "total lables of the overall dataset"}
    )

    def build_model(self):
        bert_config = BertConfig.from_pretrained(
            self.hyperparameters['bert_pretrained'])
        model = BertForMaskedLM.from_pretrained(
            self.hyperparameters['bert_pretrained'], config=bert_config)
        
        model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(
            self.num_labels, self.hyperparameters['bert_embedding_dim'])
        
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(
            mean=self.hyperparameters['bert_weight_norm_mean'], 
            std=self.hyperparameters['bert_weight_norm_std'])
        
        model.cuda()
        device_ids = [idx for idx in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
            
        return model
