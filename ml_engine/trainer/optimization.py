from typing import List
from dataclasses import dataclass, field
from transformers import AdamW, get_linear_schedule_with_warmup


@dataclass(eq=False)
class CbertOptimizer:
    model_params: List = field(
        default=None, metadata={"help": "model parameters for list type"}
    )
    num_warmup_steps: int = field(
        default=None, metadata={"help": "total num_warmup_steps which 0.1 of total steps "}
    )
    num_training_steps: int = field(
        default=None, metadata={"help": "total num_training_steps"}
    )
    optimizer_parameters: dict = field(
        default=None, metadata={"help": "about optimizer config with dict type"}
    )
    
    def __post_init__(self):
        self.optimizer_grounded_parameters = [
            {'params': [p for n, p in self.model_params if not any(nd in n for nd in self.optimizer_parameters['no_decay_weights'])], 
             'weight_decay_rate': self.optimizer_parameters['weight_decay_rate']},
            {'params': [p for n, p in self.model_params if any(nd in n for nd in self.optimizer_parameters['no_decay_weights'])], 
             'weight_decay_rate': self.optimizer_parameters['weight_no_decay_rate']}
        ]
    
    def get_optimizer_and_scheduler(self):
        optimizer = AdamW(self.optimizer_grounded_parameters,
                          lr=self.optimizer_parameters['learning_rate'], 
                          correct_bias=False)
        
        return optimizer
