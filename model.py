import torch.optim as optim
import torch
import time
from transformers import BertTokenizer, BertModel
from torch import nn, zeros, cat, argmax, ones, long, empty, tensor, full
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd

class SummaryDataset(Dataset):
    def __init__(self, tokenized_doc, tokenized_sum):
        self.tokenized_doc = tokenized_doc
        self.tokenized_sum = tokenized_sum

    def __len__(self):
        return len(self.tokenized_doc['input_ids'])

    def __getitem__(self, idx):
        return (self.tokenized_doc['input_ids'][idx],self.tokenized_doc['token_type_ids'][idx], self.tokenized_doc['attention_mask'][idx]), (self.tokenized_sum['input_ids'][idx],self.tokenized_sum['token_type_ids'][idx], self.tokenized_sum['attention_mask'][idx])




class summery_encoder(nn.Module):
    def __init__(self, number_attention_heads,num_decoder_layers,summary_token_length, vocab_size):
        super().__init__()
        
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=number_attention_heads,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.linear = nn.Linear(summary_token_length * 768,vocab_size)

        self.summary_token_length = summary_token_length
        self.vocab_size = vocab_size
 

    def forward(self, tokenized_inputs, tokenized_summary):

        encoder_encoding = self.bert_model(**tokenized_inputs)[0]
        
        batch_size = encoder_encoding.size(0)

        output_encoding = {'input_ids': full((batch_size, 1), 101,dtype=long), 
                           'token_type_ids':full((batch_size, 1), 0, dtype=long),
                           'attention_mask':full((batch_size, 1), 1, dtype=long)}


        logits_predict = zeros((encoder_encoding.size(0)*(self.summary_token_length - 1),self.vocab_size))


        for i in range(self.summary_token_length - 1):
        
            draft_encoding = self.bert_model(**output_encoding)[0]
            decoder_output = self.transformer_decoder(draft_encoding,encoder_encoding)
            flatten_output = decoder_output.view(decoder_output.size(0),-1)

            masked_flatten = cat((flatten_output, zeros((flatten_output.size(0),(self.summary_token_length - i - 1) * 768))),dim=1)

            logits = self.linear(masked_flatten)
            
            for j in range(logits.size(0)):
                logits_predict[i + (j*(self.summary_token_length - 1))] = logits[j]


            output_encoding['input_ids'] = cat((output_encoding['input_ids'],(tokenized_summary['input_ids'][:,(i+1)]).unsqueeze(1)),dim=1)
            output_encoding['attention_mask'] = cat((output_encoding['attention_mask'],(tokenized_summary['attention_mask'][:,(i+1)]).unsqueeze(1)),dim=1)
            output_encoding['token_type_ids'] = cat((output_encoding['token_type_ids'],(tokenized_summary['token_type_ids'][:,(i+1)]).unsqueeze(1)),dim=1)
            

            
        return logits_predict
    

    def predict(self, tokenized_inputs):
        
        encoder_encoding = self.bert_model(**tokenized_inputs)[0]
        
        batch_size = encoder_encoding.size(0)

        output_encoding = {'input_ids': full((batch_size, 1), 101,dtype=long), 
                           'token_type_ids':full((batch_size, 1), 0, dtype=long),
                           'attention_mask':full((batch_size, 1), 1, dtype=long)}



        for i in range(self.summary_token_length - 1):
        
            draft_encoding = self.bert_model(**output_encoding)[0]
            decoder_output = self.transformer_decoder(draft_encoding,encoder_encoding)
            flatten_output = decoder_output.view(decoder_output.size(0),-1)

            masked_flatten = cat((flatten_output, zeros((flatten_output.size(0),(self.summary_token_length - i - 1) * 768))),dim=1)

            logits = self.linear(masked_flatten)
            

            probabilities = nn.functional.softmax(logits, dim=1)
            predictions = argmax(probabilities, dim=1).unsqueeze(1)


            output_encoding['input_ids'] = cat(( output_encoding['input_ids'],predictions),dim=1)
            output_encoding['attention_mask'] = cat((output_encoding['attention_mask'],ones((predictions.size(0),1), dtype=long)),dim=1)
            output_encoding['token_type_ids'] = cat((output_encoding['token_type_ids'],zeros((predictions.size(0),1), dtype=long)),dim=1)

        return output_encoding['input_ids']



def rough_predict(input_):

    return model.predict(input_)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv('cnn_dailymail/train.csv')
df.replace('', pd.NA, inplace=True)

# Drop rows with NaNs in either col1 or col2
df.dropna(inplace=True)
df = df.sample(n=2500, random_state=42)  # 30 percent -> 35,000 samples


# dataset gives out string x, string y
# have the summary length before
# Then tokenize document and tokenize y such that all lengths in the batch equal summary length

tokenized_input = tokenizer(df['article'].tolist(), return_tensors="pt", padding='max_length', max_length=512, truncation=True)
tokenized_output = tokenizer(df['highlights'].tolist(), return_tensors="pt", padding='max_length' , max_length=128, truncation=True)



summary_length = (tokenized_output['input_ids']).size(1)


data_object = SummaryDataset(tokenized_input, tokenized_output)

total_length = len(data_object)
print(total_length)

train_length = int(0.70 * total_length)
val_length = int(0.20 * total_length)
test_length = total_length - train_length - val_length

print(f"Train length: {train_length}")
print(f"Val length: {val_length}")
print(f"Test length: {test_length}")

train_dataset, val_dataset, test_dataset = random_split(data_object, [train_length, val_length, test_length])

batch_size = 32
Epochs = 3

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

del df
del data_object
del train_dataset
del val_dataset
del test_dataset


training_losses = []
validation_losses = []
testing_losses = []


model = summery_encoder(number_attention_heads=6,num_decoder_layers=8,summary_token_length=summary_length,vocab_size=tokenizer.vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

print("Document Length: " + str(len(tokenized_input['input_ids'][0])))
print("Summary Length: " + str(summary_length))



#----------------------------------------------------------------------------------------
start_time = time.time()
step_size = 10

for epoch in range(Epochs):

    print(f"Epoch: {epoch}")
    print("-------------------------training-------------------------")
    model.train()
    total_loss = 0
    len_data = len(train_dataloader)


    print(f"{len_data} training batches")
    for i, batch in enumerate(train_dataloader):
        tok_input,tok_output = batch[0],batch[1]

        tok_input = {'input_ids':tok_input[0], 'token_type_ids':tok_input[1], 'attention_mask':tok_input[2]}
        tok_output = {'input_ids':tok_output[0], 'token_type_ids':tok_output[1], 'attention_mask':tok_output[2]}

        logits = model(tok_input, tok_output)

        tok_con_output = (tok_output['input_ids'][:, 1:]).clone()
        tok_con_output = tok_con_output.view(-1)


        loss = loss_function(logits,tok_con_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tok_con_output = None
        del tok_con_output

        if i % (len_data // step_size) == 0:
            print(loss.item())
        print("---------------------Batch: "+ str(i) + " ------------------------")

    average_loss = total_loss/len_data
    training_losses.append(average_loss)
    print(f"Average Training Loss: {average_loss}")






    print()
    print("-------------------------validation-------------------------")
    model.eval()
    total_loss = 0
    len_data = len(val_dataloader)

    print(f"{len_data} validation batches")
    with torch.no_grad():
      for i, batch in enumerate(val_dataloader):
        tok_input,tok_output = batch[0],batch[1]

        tok_input = {'input_ids':tok_input[0], 'token_type_ids':tok_input[1], 'attention_mask':tok_input[2]}
        tok_output = {'input_ids':tok_output[0], 'token_type_ids':tok_output[1], 'attention_mask':tok_output[2]}

        logits = model(tok_input, tok_output)

        tok_con_output = (tok_output['input_ids'][:, 1:]).clone()
        tok_con_output = tok_con_output.view(-1)


        loss = loss_function(logits,tok_con_output)


        tok_con_output = None
        del tok_con_output

        if i % (len_data // step_size) == 0:
            print(loss.item())
        print("---------------------Batch: "+ str(i) + " ------------------------")

            
    average_loss = total_loss/len_data
    validation_losses.append(average_loss)
    print(f"Average Validation Loss: {average_loss}")






    print()
    print("-------------------------testing-------------------------")
    total_loss = 0
    len_data = len(test_dataloader)

    print(f"{len_data} testing batches")
    with torch.no_grad():
      for i, batch in enumerate(test_dataloader):
        tok_input,tok_output = batch[0],batch[1]

        tok_input = {'input_ids':tok_input[0], 'token_type_ids':tok_input[1], 'attention_mask':tok_input[2]}
        tok_output = {'input_ids':tok_output[0], 'token_type_ids':tok_output[1], 'attention_mask':tok_output[2]}

        logits = model(tok_input, tok_output)

        tok_con_output = (tok_output['input_ids'][:, 1:]).clone()
        tok_con_output = tok_con_output.view(-1)


        loss = loss_function(logits,tok_con_output)


        tok_con_output = None
        del tok_con_output

        if i % (len_data // step_size) == 0:
            print(loss.item())
        print("---------------------Batch: "+ str(i) + " ------------------------")


    average_loss = total_loss/len_data
    testing_losses.append(average_loss)
    print(f"Average Testing Loss: {average_loss}")
 
    model.save_pretrained(f"saved_models/test_model_{epoch}")
    print()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print()

#----------------------------------------------------------------------------------------


print(training_losses)
print(validation_losses)
print(testing_losses)


'''
        if i % (len_data // step_size) == 0:
            print(loss.item())
            print(rough_predict(tok_input))
            print([tokenizer.decode(j) for j in rough_predict(tok_input)])
            print(f"Actual: \n {[tokenizer.decode(tok_output['input_ids'][j]) for j in range(len(tok_output['input_ids'])) ]}")
            print("---------------------Batch: "+ str(i) + " ------------------------")
'''




# move on to thisdepending on results of first.

# then take full summary draft and mask one word linearly so <MASK> is awesome, She <MASK> awesome
# feed this mask into the BERt and send these embeddings to decoder with same document embeddings to get output probability for however long we want the sequence length
#(note final output probability is not fed back into model)
