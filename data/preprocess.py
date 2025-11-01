import torch

class SFTDataProcessor:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer      
        self.IGNORE_TOKEN_ID = -100
        self.max_len = max_len
        self.system_message = "You are a helpful assistant."

    def __call__(self, sources, max_len=None):
        tokenizer = self.tokenizer
        IGNORE_TOKEN_ID = self.IGNORE_TOKEN_ID
        im_start = tokenizer.bos_token_id
        im_end = tokenizer.eos_token_id
        nl_tokens = tokenizer('\n').input_ids
        _system = tokenizer('system').input_ids + nl_tokens
        _user = tokenizer('user').input_ids + nl_tokens
        _assistant = tokenizer('assistant').input_ids + nl_tokens
        roles = {"user":"<|im_start|>user", "assistant":"<|im_start|>assistant"}


        if max_len is None:
            max_len = self.max_len

        input_ids, labels = [], []
        
        for i, msg in enumerate(sources["conversations"]):

            input_id, label = [], []
            system = [im_start] + _system + tokenizer(self.system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            label += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
            assert len(input_id) == len(label)

            for j, sentence in enumerate(msg):
                role = roles[sentence['role']]
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence['content']).input_ids + [im_end] + nl_tokens
                input_id += _input_id
                if role == roles['user']:
                    label += [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
                elif role == roles['assistant']:
                    label += [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError
                
                assert len(input_id) == len(label)
            
            if len(input_id) > max_len:
                input_id = input_id[:max_len]
                label = label[:max_len]

            input_ids.append(input_id)
            labels.append(label)

        return dict(input_ids=input_ids, labels=labels)

class OldPretrainDataProcessor:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, sources, max_len=None):
        if max_len is None:
            max_len = self.max_len

        tokenized_source = self.tokenizer(
            sources["text"],
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )

        input_ids = tokenized_source["input_ids"]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = input_ids.clone()
        attention_mask = tokenized_source["attention_mask"]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return dict(input_ids=input_ids, labels=target_ids, attention_mask=attention_mask)



class PretrainDataProcessor:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, sources, max_len=None):
        if max_len is None:
            max_len = self.max_len
        tokenized_source = self.tokenizer(
            sources["text"],
            max_length=max_len,
            padding=False,
            truncation=True,
        )
        
        input_ids = tokenized_source["input_ids"]
        return dict(input_ids=input_ids)


class ProcessorForComputeTokens:
    def __init__(self, tokenizer, max_len=2048):
        self.tokenizer = tokenizer

    def __call__(self, sources, max_len=2048):
        tokenized_source = self.tokenizer(
            sources["text"],
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

        token_length = [len(ids) for ids in tokenized_source["input_ids"]]
        return {"token_length": token_length}


    
class ProcessorForChatML:

    system_prompt = "You are a helpful assistant."
    
    def __call__(self, sources):
        
        processed_batch = [] 
        
        for messages in sources['conversations']:
            
            message_storage = []
            message_storage.append({'role': 'system', 'content': self.system_prompt}) 
            
            for message_dict in messages:
                message_storage.append(message_dict) 
            
            processed_batch.append(message_storage)


        return {'messages': processed_batch}

        


