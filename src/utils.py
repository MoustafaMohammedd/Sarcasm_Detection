import copy
import matplotlib.pyplot as plt
import torch
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")

def prepare_data (sentence):
  tokens=tokenizer(sentence)
  tokens_l=[]
  for token in tokens :
    if token.isalnum()== False :
      continue
    tokens_l.append(token.lower())

  return tokens_l


def pad_sentence (sentence_l,max_len,pad_token):
  if len(sentence_l)>=max_len:
    return sentence_l[:max_len]
  else:
    return sentence_l + [pad_token] * (max_len-len(sentence_l))


def our_accuracy (target,pred): 
  target=target.squeeze() 
  pred=torch.round(torch.sigmoid(pred.squeeze()))
  n_correct = sum(target==pred).item()
  return  n_correct




class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.best_accuracy = None  # New attribute to track best accuracy
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss, val_accuracy):
        if self.best_loss is None or val_loss < self.best_loss:  # Update if val_loss is better
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy  # Update best accuracy along with best loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}. " \
                          # f"Best Loss: {self.best_loss:.4f}, Best Accuracy: {self.best_accuracy:.2f}%"

        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}. " \
                          # f"Best Loss: {self.best_loss:.4f}, Best Accuracy: {self.best_accuracy:.2f}%"

        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs. " \
                          # f"Best Loss: {self.best_loss:.4f}, Best Accuracy: {self.best_accuracy:.2f}%"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs. " \
                              f"Best Loss: {self.best_loss:.4f}, Best Accuracy: {self.best_accuracy:.2f}%"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


def save_checkpoint(model, optimizer, epoch, loss, filename="model_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)


def plot_l_a(train_loss,test_loss,train_accuracy,test_accuracy,name): 
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    axs[0].plot(train_loss, label='Training Loss')
    axs[0].plot(test_loss, label='Test Loss')
    axs[0].set_title('Training and Test Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    #axs[0].set_ylim([0, 1])
    axs[0].legend()
    
    axs[1].plot(train_accuracy, label='Training Accuracy')
    axs[1].plot(test_accuracy, label='Test Accuracy')
    axs[1].set_title('Training and Test Accuracy over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    #axs[1].set_ylim([0, 100])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig (r'D:\Sarcasm_Detection\images' + f"\{name}")
    plt.show()
    
    
def our_predict(s, our_model, our_vocab,config_hp, device):
   
    l=prepare_data(s)
   
    p_l=pad_sentence(l,config_hp["MAX_LEN"],config_hp["SPECIAL_TOKENS"][0])
   
    p_l=torch.tensor(our_vocab.lookup_indices(p_l),dtype=torch.long).reshape(1,-1).to(device)
    
    pred=our_model(p_l)
    
    pred=torch.round(torch.sigmoid(pred))
    pred=pred.cpu().detach().numpy().item()
    return pred
    