from transformers import TrainingArguments, Trainer,DataCollatorWithPadding 

class BaseModel(Trainer): 
     def __init__(self, model, args, train_dataset, eval_dataset, tokenizer): 
         self.model = model 
         self.args = args 
         self.train_dataset = train_dataset 
         self.eval_dataset = eval_dataset 
         self.tokenizer = tokenizer 
         self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer) 
         super().__init__(model=self.model, args=self.args, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, data_collator=self.data_collator) 
     def train(self): 
         self.train() 
     def evaluate(self): 
         self.evaluate()
     