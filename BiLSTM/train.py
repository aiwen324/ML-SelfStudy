import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import Constants
from dataset import TranslationDataset, paired_collate_fn
from model import Transformer


def cal_performance(pred, gold, smoothing=False):
    pred = pred.contiguous().view(-1, pred.shape[2])
    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]
    # print(pred)
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def cal_loss(pred, gold, smoothing):
    # print(gold.shape)
    gold = gold.contiguous().view(-1)
    # print(gold.shape)
    # print(pred.shape)
    # print(gold)
    loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss
        



def train_epoch(model, training_data, optimizer, device):

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
        training_data, mininterval=2,
        desc='  -  (Training)   ', leave=False, ascii=True):

        # prepare data
        src_seq, tgt_seq = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        src_max_len = src_seq.shape[1]
        tgt_max_len = tgt_seq.shape[1]
        pred = model(src_seq, src_max_len, tgt_max_len)
        # print('## debug msg: checking gold shape and pred shape')
        # print(gold.shape, pred.shape)

        # backward
        loss, n_correct = cal_performance(pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total

    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False, ascii=True):

            # prepare data
            src_seq, tgt_seq = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            src_max_len, tgt_max_len = src_seq.shape[1], tgt_seq.shape[1]
            pred = model(src_seq, src_max_len, tgt_max_len)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device)
        print('  - (Trainig)    ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i
        }

        model_name = 'best_model.chkpt'
        if valid_accu >= max(valid_accus):
            torch.save(checkpoint, model_name)
            print('     - [Info] The checkpoint file has been updated.')



def prepare_dataloaders(data, opt):
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader


def main():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('-data', required=True)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # Load data
    data = torch.load(opt.data)

    opt.max_token_seq_len = data['settings'].max_word_seq_len + 2

    training_data, validation_data = prepare_dataloaders(data, opt)
    
    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    print(opt)
    # opt.cuda = True
    device = torch.device('cuda' if opt.cuda else 'cpu')

    # TODO: Fill the code
    transformer = Transformer(
        d_word_embedding=opt.d_word_vec,
        d_h=opt.d_model,
        d_s=opt.d_model,
        src_vocab_size=opt.src_vocab_size,
        tgt_vocab_size=opt.tgt_vocab_size,
        max_sent_len=opt.max_token_seq_len).to(device)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, transformer.parameters()),
        betas=(0.9, 0.98), eps=1e-09)

    train(transformer, training_data, validation_data, optimizer, device, opt)

if __name__ == '__main__':
    main()