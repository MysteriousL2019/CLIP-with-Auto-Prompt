import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.optim as optim
from tqdm.autonotebook import tqdm
from collections import OrderedDict
import torch.nn.functional as F
from torch.optim import lr_scheduler
import netron
import torch.onnx
from torch.autograd import Variable

_tokenizer = _Tokenizer()

def setups():
    os.system('pip install openai-clip')

def build_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load('ViT-B/32', device)

    return model, preprocess, torch.device(device)

def define_input(ctx_init: str):
    ctx_init = ctx_init.replace("_", " ")
    n_ctx = len(ctx_init.split(" "))
    prompt_prefix = ctx_init
    print(f'Initial context: "{prompt_prefix}"')
    print(f"Number of context words (tokens): {n_ctx}")
    return prompt_prefix, n_ctx

def get_tokens(prompt_prefix:str,  classnames:list, device: torch.device):
  classnames = [name.replace("_", " ") for name in classnames]
  name_lens = [len(_tokenizer.encode(name)) for name in classnames]
  prompts = [prompt_prefix + " " + name + "." for name in classnames]

  tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
  with torch.no_grad():
    embedding = model.token_embedding(tokenized_prompts).type(model.dtype)  # (n_cls, n_tkn, dim)

  token_prefix = nn.Parameter(embedding[:, :1, :], requires_grad=False).to(device)  # (n_cls, 1, dim)
  token_suffix = nn.Parameter(embedding[:, 1 + n_ctx:, :], requires_grad=False).to(device)  # (n_cls, 1, dim)
  trainable_hidden_state = nn.Parameter(embedding[:, 1:1 + n_ctx,:].clone(),requires_grad=True).to(device)
  return token_prefix, token_suffix, tokenized_prompts, trainable_hidden_state

class MetaNet(nn.Module):
    def __init__(self, model):
        super(MetaNet, self).__init__()
        
        self.fc1 = nn.Linear(model.visual.output_dim, model.visual.output_dim // 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(model.visual.output_dim // 16, model.ln_final.weight.shape[0])

    def forward(self, x):
        x = self.fc1(x)
        self.relu(x)
        return self.fc2(x)

def construct_prompts(im_features,token_prefix,token_suffix,trainable_hidden_state,len_classes):
  bias = meta_net(im_features).to(device) # (batch, ctx_dim)
  bias = bias.unsqueeze(1).to(device)   # (batch, 1, ctx_dim) 16,1,512
  trainable_hidden_state = trainable_hidden_state.mean(dim=0, keepdim=True) # 
  trainable_hidden_state_shifted = trainable_hidden_state + bias           
  prompts = []
  for hidden_shifted_i in trainable_hidden_state_shifted:
      hid_i = hidden_shifted_i.unsqueeze(0).expand(len_classes, -1, -1)
      pts_i = torch.cat((token_prefix, hid_i, token_suffix),dim=1)
      prompts.append(pts_i)
  prompts = torch.stack(prompts).to(device)
  return prompts


def text_encoder(prompt_all,tokenized_prompts):
  x = prompt_all + model.positional_embedding.type(model.dtype)
  x = x.permute(1, 0, 2)  # NLD -> LND
  x = model.transformer(x)
  x = x.permute(1, 0, 2)  # LND -> NLD
  x = model.ln_final(x).type(model.dtype)

  # take features from the eot embedding (eot_token is the highest number in each sequence)
  x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ model.text_projection
  return x


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
    

def forward_(images_batch, labels_batch=None):
    image_features = model.encode_image(images_batch)
    image_features_norm = image_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features_norm
    prompt_all = construct_prompts(image_features,token_prefix,token_suffix,trainable_hidden_state,len(cifar100_data.classes))
    similarity_list = []
    for pts_i, imf_i in zip(prompt_all,image_features):
        text_features = text_encoder(pts_i,tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity_list.append(similarity)
    similarity_list = torch.stack(similarity_list)
    if model.train and labels_batch!=None:
        return F.cross_entropy(similarity_list.mean(dim=0), labels_batch)
    else:
        return similarity_list


if __name__=='__main__':
    setups()
    model, preprocess, device = build_model()
    cifar100_data = CIFAR100(root=os.path.join(os.getcwd(), 'dataset'), train=False, transform=preprocess,download=True)
    prompt_prefix, n_ctx = define_input('a photo of a')
    token_prefix, token_suffix, tokenized_prompts, trainable_hidden_state = get_tokens(prompt_prefix, cifar100_data.classes)
    
    meta_net = MetaNet(model).to(device)
    optimizer = optim.SGD(list(meta_net.parameters()) + [trainable_hidden_state], lr=0.0001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    dataloader = torch.utils.data.DataLoader(cifar100_data, batch_size=16, shuffle=True)
num_of_epoch = 20
for epoch in range(num_of_epoch):
    model.train()
    scheduler.step()
    avg_loss = AvgMeter()
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_of_epoch}')
    for images_batch, labels_batch in progress_bar:
        images_batch = images_batch.to(device) # [batch_size, 3, 224, 224]
        labels_batch = labels_batch.to(device)

        loss = forward_(images_batch, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        progress_bar.set_postfix({'loss': avg_loss.avg})
    torch.save(meta_net.state_dict(), f'{os.getcwd()}/CoCoOp_model/meta_net_{epoch+1}.pth')
    torch.save(trainable_hidden_state, f'{os.getcwd()}/CoCoOp_model/trainable_hidden_state_{epoch+1}.pth')
    progress_bar.close()
  #     pbar.update(1)
    print('Epoch:', epoch+1, 'Loss:', avg_loss.avg)
