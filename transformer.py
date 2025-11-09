import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# 确保下载必要的NLTK资源
nltk.download('punkt', quiet=True)

# ----------------------------
# Transformer核心组件实现（精简验证输出）
# ----------------------------

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        # 仅保留初始化成功提示
        print(f"✅ 位置编码初始化完成（d_model={d_model}）")

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # 仅保留初始化成功提示
        print(f"✅ 多头注意力初始化完成（num_heads={num_heads}）")

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(output), attn


class PositionWiseFeedForward(nn.Module):
    """位置-wise前馈网络"""
    def __init__(self, d_model, dff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        # 仅保留初始化成功提示
        print(f"✅ FFN初始化完成（d_model={d_model} → dff={dff}）")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Encoder(nn.Module):
    """完整编码器"""
    def __init__(self, input_vocab_size, d_model, num_layers, num_heads, dff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])
        # 仅保留核心配置提示
        print(f"✅ 编码器初始化完成（num_layers={num_layers}, vocab_size={input_vocab_size}）")

    def forward(self, x, mask):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x * np.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x


class Decoder(nn.Module):
    """完整解码器"""
    def __init__(self, target_vocab_size, d_model, num_layers, num_heads, dff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])
        # 仅保留核心配置提示
        print(f"✅ 解码器初始化完成（num_layers={num_layers}, vocab_size={target_vocab_size}）")

    def forward(self, x, enc_output, self_mask, cross_mask):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x * np.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)
            
        return x


class Transformer(nn.Module):
    """完整Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, 
                 num_heads=8, dff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(
            input_vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff,
            max_len=max_len,
            dropout=dropout
        )
        self.decoder = Decoder(
            target_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff,
            max_len=max_len,
            dropout=dropout
        )
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        # 仅保留核心配置提示
        print(f"✅ Transformer模型初始化完成（d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}）")

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, cross_mask)
        final_output = self.final_layer(dec_output)
        
        return final_output


# ----------------------------
# 掩码函数（删除冗余形状打印）
# ----------------------------

def create_padding_mask(seq, pad_idx):
    """创建填充掩码，与输入序列同设备"""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask


def create_look_ahead_mask(seq_len, device):
    """创建前瞻掩码，强制指定设备"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    """创建所有掩码，确保100%设备一致"""
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)
    device = src.device  # 统一使用src的设备
    
    src_mask = create_padding_mask(src, src_pad_idx)
    cross_mask = create_padding_mask(src, src_pad_idx)
    tgt_pad_mask = create_padding_mask(tgt, tgt_pad_idx)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_seq_len, device).unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_pad_mask | tgt_look_ahead_mask
    
    
    
    return src_mask, tgt_mask, cross_mask


# ----------------------------
# 数据处理（精简样本和编码打印）
# ----------------------------

def parse_train_file(file_path):
    """解析训练文件，提取<doc>内非标签行"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = []
    in_doc = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('<doc'):
            in_doc = True
            continue
        if line.startswith('</doc'):
            in_doc = False
            continue
        if in_doc and not line.startswith('<'):
            sentences.append(line)
    
    # 仅保留数量提示，删除前3条样本打印
    print(f"✅ 从 {os.path.basename(file_path)} 提取出 {len(sentences)} 条句子")
    return sentences


def parse_xml_file(file_path):
    """解析XML文件，提取<seg>标签文本"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"XML解析错误 {file_path}: {e}")
        return []
    
    sentences = []
    for seg in root.iter('seg'):
        if seg.text:
            sentences.append(seg.text.strip())
    
    # 仅保留数量提示，删除前3条样本打印
    print(f"✅ 从 {os.path.basename(file_path)} 提取出 {len(sentences)} 条句子")
    return sentences


def preprocess_text(text, lang='en'):
    """仅基础预处理：小写+去除特殊字符"""
    text = text.lower()
    if lang == 'de':
        text = re.sub(r"[^a-zA-Z0-9äöüßàâäèéêëîïôöùûüÿç\s]", " ", text)
    else:
        text = re.sub(r"[^a-zA-Z0-9àâäèéêëîïôöùûüÿç\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Vocabulary:
    def __init__(self, max_size=10000):
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        
        self.token2idx = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.max_size = max_size
        self.word_count = Counter()
    
    def update(self, sentence):
        if not sentence:
            return
        tokens = sentence.split()
        self.word_count.update(tokens)
    
    def build(self):
        most_common = self.word_count.most_common(self.max_size - len(self.token2idx))
        for word, _ in most_common:
            idx = len(self.token2idx)
            self.token2idx[word] = idx
            self.idx2token[idx] = word
        
        # 保留核心信息，删除冗余打印
        print(f"✅ 词汇表构建完成（大小={len(self.token2idx)}）")
        print(f"🔍 高频词Top5：{list(self.word_count.most_common(5))}")
    
    def encode(self, sentence, max_length=None):
        if not sentence:
            sentence = self.unk_token
        tokens = sentence.split()
        encoded = [self.token2idx[self.sos_token]]
        encoded += [self.token2idx.get(token, self.token2idx[self.unk_token]) for token in tokens]
        encoded += [self.token2idx[self.eos_token]]
        if max_length is not None:
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            else:
                encoded += [self.token2idx[self.pad_token]] * (max_length - len(encoded))
        # 删除每个句子的编码打印（避免刷屏）
        return encoded
    
    def decode(self, indices):
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, self.unk_token)
            if token == self.eos_token:
                break
            if token not in [self.pad_token, self.sos_token]:
                tokens.append(token)
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.token2idx)


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        assert len(src_sentences) == len(tgt_sentences), "源/目标语言句子数不匹配"
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        src_encoded = self.src_vocab.encode(src_sentence, self.max_len)
        tgt_encoded = self.tgt_vocab.encode(tgt_sentence, self.max_len)
        
        src_len = min(len(src_sentence.split()) + 2, self.max_len)
        tgt_len = min(len(tgt_sentence.split()) + 2, self.max_len)
        
        # 删除每10000个样本的打印（避免刷屏）
        return {
            'src': torch.tensor(src_encoded, dtype=torch.long),
            'tgt': torch.tensor(tgt_encoded, dtype=torch.long),
            'src_len': torch.tensor(src_len, dtype=torch.long),
            'tgt_len': torch.tensor(tgt_len, dtype=torch.long)
        }


def load_iwslt_dataset(data_dir, src_lang, tgt_lang, max_len=50, max_vocab_size=10000):
    """加载数据集"""
    prefix = f"{src_lang}-{tgt_lang}"
    train_src_path = os.path.join(data_dir, f'train.tags.{prefix}.{src_lang}')
    train_tgt_path = os.path.join(data_dir, f'train.tags.{prefix}.{tgt_lang}')
    dev_src_path = os.path.join(data_dir, f'IWSLT17.TED.dev2010.{prefix}.{src_lang}.xml')
    dev_tgt_path = os.path.join(data_dir, f'IWSLT17.TED.dev2010.{prefix}.{tgt_lang}.xml')
    test_src_path = os.path.join(data_dir, f'IWSLT17.TED.tst2010.{prefix}.{src_lang}.xml')
    test_tgt_path = os.path.join(data_dir, f'IWSLT17.TED.tst2010.{prefix}.{tgt_lang}.xml')
    
    # 检查文件存在性
    for path in [train_src_path, train_tgt_path, dev_src_path, dev_tgt_path, test_src_path, test_tgt_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集文件不存在: {path}")
    
    # 加载原始数据
    print("\n=== 加载训练数据 ===")
    train_src_raw = parse_train_file(train_src_path)
    train_tgt_raw = parse_train_file(train_tgt_path)
    print("\n=== 加载开发数据 ===")
    dev_src_raw = parse_xml_file(dev_src_path)
    dev_tgt_raw = parse_xml_file(dev_tgt_path)
    print("\n=== 加载测试数据 ===")
    test_src_raw = parse_xml_file(test_src_path)
    test_tgt_raw = parse_xml_file(test_tgt_path)
    
    # 基础预处理
    def basic_preprocess(src_sents, tgt_sents, src_lang, tgt_lang):
        processed_src = [preprocess_text(sent, src_lang) for sent in src_sents]
        processed_tgt = [preprocess_text(sent, tgt_lang) for sent in tgt_sents]
        return processed_src, processed_tgt
    
    print("\n=== 基础预处理 ===")
    train_src, train_tgt = basic_preprocess(train_src_raw, train_tgt_raw, src_lang, tgt_lang)
    dev_src, dev_tgt = basic_preprocess(dev_src_raw, dev_tgt_raw, src_lang, tgt_lang)
    test_src, test_tgt = basic_preprocess(test_src_raw, test_tgt_raw, src_lang, tgt_lang)
    
    # 数据量验证（保留核心）
    print(f"\n数据量验证:")
    print(f"训练集: 源{len(train_src)}条 | 目标{len(train_tgt)}条")
    print(f"开发集: 源{len(dev_src)}条 | 目标{len(dev_tgt)}条")
    print(f"测试集: 源{len(test_src)}条 | 目标{len(test_tgt)}条")
    
    # 构建词汇表
    print("\n=== 构建词汇表 ===")
    src_vocab = Vocabulary(max_size=max_vocab_size)
    tgt_vocab = Vocabulary(max_size=max_vocab_size)
    print(f"🔍 正在更新源语言词汇表...")
    for sent in train_src[:10000]:  # 先更新前10000条加速验证
        src_vocab.update(sent)
    print(f"🔍 正在更新目标语言词汇表...")
    for sent in train_tgt[:10000]:
        tgt_vocab.update(sent)
    src_vocab.build()
    tgt_vocab.build()
    
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len)
    dev_dataset = TranslationDataset(dev_src, dev_tgt, src_vocab, tgt_vocab, max_len)
    test_dataset = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len)
    
    print(f"✅ 数据集创建完成（train={len(train_dataset)}, dev={len(dev_dataset)}, test={len(test_dataset)}）")
    return {
        'train': train_dataset,
        'dev': dev_dataset,
        'test': test_dataset,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }


# ----------------------------
# 训练和评估函数（精简批次打印）
# ----------------------------

def train_model(
    model, train_loader, dev_loader, src_vocab, tgt_vocab,
    epochs=10, lr=1e-4, device='cuda', model_save_path='transformer_iwslt_en_de.pth'
):
    if torch.cuda.device_count() > 1:
        print(f"⚠️  检测到 {torch.cuda.device_count()} 块GPU，使用DataParallel加速")
        model = nn.DataParallel(model)
    
    pad_idx = src_vocab.token2idx[src_vocab.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    model.to(device)
    train_losses = []
    dev_losses = []
    best_bleu = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        print(f"\n=== Epoch {epoch+1}/{epochs} 训练开始 ===")
        for batch_idx, batch in enumerate(train_loader):
            # 所有数据移到设备
            src = batch['src'].to(device, non_blocking=True)
            tgt = batch['tgt'].to(device, non_blocking=True)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            
            # 创建掩码
            src_mask, tgt_mask, cross_mask = create_masks(
                src, tgt_input, 
                src_vocab.token2idx[src_vocab.pad_token],
                tgt_vocab.token2idx[tgt_vocab.pad_token]
            )
            
            # 前向传播
            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask, cross_mask)
            
            # 计算损失
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * src.size(0)
            

        
        # 平均损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 开发集评估
        dev_loss, dev_bleu = evaluate(model, dev_loader, src_vocab, tgt_vocab, criterion, device)
        dev_losses.append(dev_loss)
        
        # 学习率调度
        scheduler.step(dev_loss)
        
        # 保存最佳模型
        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            print(f"✅ 保存最佳模型（BLEU: {best_bleu:.4f}）")
        
        # 打印日志
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs} 结束")
        print(f"train_loss: {train_loss:.4f} | val_loss: {dev_loss:.4f} | BLEU: {dev_bleu:.4f} | 时间: {epoch_time:.2f}秒")
    
    plt.plot(train_losses, label='train_loss')
    plt.plot(dev_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_curve_en_de.png')
    plt.close()
    print(f"✅ 损失曲线已保存为 loss_curve_en_de.png")
    
    return model


def evaluate(model, dataloader, src_vocab, tgt_vocab, criterion, device):
    model.eval()
    total_loss = 0.0
    all_references = []
    all_hypotheses = []
    smoothing = SmoothingFunction().method4
    pad_idx = src_vocab.token2idx[src_vocab.pad_token]
    
    print("\n=== 开始评估 ===")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            src = batch['src'].to(device, non_blocking=True)
            tgt = batch['tgt'].to(device, non_blocking=True)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 仅第1批次打印形状
            if batch_idx == 0:
                print(f"🔍 评估第1批次形状：src={src.shape}, tgt_output={tgt_output.shape}")
            
            src_mask, tgt_mask, cross_mask = create_masks(
                src, tgt_input, 
                src_vocab.token2idx[src_vocab.pad_token],
                tgt_vocab.token2idx[tgt_vocab.pad_token]
            )
            
            logits = model(src, tgt_input, src_mask, tgt_mask, cross_mask)
            
            # 计算损失
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            total_loss += loss.item() * src.size(0)
            
            # 收集预测结果
            preds = torch.argmax(logits, dim=-1)
            # 仅第1批次打印前1个样本对比
            if batch_idx == 0:
                reference = tgt_vocab.decode(tgt_output[0].cpu().numpy())
                hypothesis = tgt_vocab.decode(preds[0].cpu().numpy())
                print(f"🔍 评估样本示例：")
                print(f"  参考：{reference}")
                print(f"  预测：{hypothesis}")
            
            for i in range(src.size(0)):
                reference = tgt_vocab.decode(tgt_output[i].cpu().numpy())
                all_references.append([reference.split()])
                hypothesis = tgt_vocab.decode(preds[i].cpu().numpy())
                all_hypotheses.append(hypothesis.split())
    
    avg_loss = total_loss / len(dataloader.dataset)
    bleu_score = sum(sentence_bleu(ref, hyp, smoothing_function=smoothing) 
                     for ref, hyp in zip(all_references, all_hypotheses)) / len(all_references)
    
    print(f"✅ 评估完成（平均损失={avg_loss:.4f}, BLEU={bleu_score*100:.4f}）")
    return avg_loss, bleu_score * 100


# ----------------------------
# 束搜索解码（精简中间步骤打印）
# ----------------------------

def translate_beam_search(
    model, sentence, src_vocab, tgt_vocab, 
    src_lang='en', tgt_lang='de', max_len=50, device='cuda',
    beam_size=5, repeat_penalty=1.2, temperature=0.7
):
    """使用束搜索解码，避免重复翻译（精简输出）"""
    model.eval()
    if hasattr(model, 'module'):
        model = model.module  # 多GPU模型适配
    
    # 预处理输入句子
    processed = preprocess_text(sentence, src_lang)
    print(f"🔍 翻译输入：{sentence} → 预处理后：{processed}")
    
    src_encoded = src_vocab.encode(processed, max_len)
    src_tensor = torch.tensor([src_encoded], dtype=torch.long).to(device)
    src_mask = create_padding_mask(src_tensor, src_vocab.token2idx[src_vocab.pad_token])
    
    # 提前计算编码器输出
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
    
    # 束初始化
    beams = [([tgt_vocab.token2idx[tgt_vocab.sos_token]], 0.0, 1)]
    finished = []
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            if not beams or len(finished) >= beam_size:
                break
            
            new_beams = []
            for seq, score, length in beams:
                if seq[-1] == tgt_vocab.token2idx[tgt_vocab.eos_token]:
                    finished.append((seq, score / length))
                    continue
                
                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                tgt_mask = create_look_ahead_mask(len(seq), device).unsqueeze(0)
                cross_mask = src_mask
                
                dec_output = model.decoder(tgt_tensor, enc_output, tgt_mask, cross_mask)
                logits = model.final_layer(dec_output)
                
                next_token_logits = logits[:, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token_log_probs = torch.log(next_token_probs)
                
                # 重复惩罚
                for idx in seq:
                    if idx != tgt_vocab.token2idx[tgt_vocab.sos_token]:
                        next_token_log_probs[0][idx] -= np.log(repeat_penalty)
                
                top_log_probs, top_indices = next_token_log_probs.topk(beam_size)
                for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                    new_seq = seq.copy()
                    new_seq.append(idx.item())
                    new_score = score + log_prob.item()
                    new_beams.append((new_seq, new_score, length + 1))
            
            # 保留得分最高的beam_size个束
            new_beams.sort(key=lambda x: x[1] / x[2], reverse=True)
            beams = new_beams[:beam_size]
    
    # 合并完成的序列和未完成的束
    finished.extend([(seq, score / length) for seq, score, length in beams])
    finished.sort(key=lambda x: x[1], reverse=True)
    best_seq = finished[0][0] if finished else beams[0][0]
    best_translated = tgt_vocab.decode(best_seq)
    
    # 仅打印最终候选（前2个）
    print(f"🔍 解码完成，最佳翻译：{best_translated}")
    if len(finished) > 1:
        second_translated = tgt_vocab.decode(finished[1][0])
        print(f"🔍 候选翻译2：{second_translated}")
    
    return best_translated


# ----------------------------
# 主函数（保持核心配置）
# ----------------------------

def main():
    # 配置参数
    data_dir = "./iwslt17_data"
    src_lang = "en"
    tgt_lang = "de"
    max_len = 50
    max_vocab_size = 20000
    batch_size = 64
    d_model = 512
    num_layers = 4
    num_heads = 8
    dff = 2048
    epochs = 30  # 改为30个epoch确保收敛
    lr = 1e-4
    model_save_path = "transformer_iwslt_en_de_4090.pth"
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"===== 初始化配置 =====")
    print(f"使用设备: {device} | GPU数量: {torch.cuda.device_count()} 块")
    print(f"训练语言对: {src_lang} → {tgt_lang} | 批次大小: {batch_size} | 训练轮次: {epochs}")
    print(f"======================\n")
    
    # 加载数据集
    print("\n=== 加载IWSLT 2017数据集 ===")
    try:
        dataset = load_iwslt_dataset(
            data_dir, src_lang, tgt_lang, 
            max_len=max_len, 
            max_vocab_size=max_vocab_size
        )
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset['train'], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    dev_loader = DataLoader(
        dataset['dev'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset['test'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    
    # 打印数据加载器信息
    print(f"\n=== 数据加载器创建完成 ===")
    print(f"训练集批次数量：{len(train_loader)} | 开发集：{len(dev_loader)} | 测试集：{len(test_loader)}")
    
    # 初始化模型
    print("\n=== 初始化Transformer模型 ===")
    model = Transformer(
        src_vocab_size=len(dataset['src_vocab']),
        tgt_vocab_size=len(dataset['tgt_vocab']),
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dff=dff,
        max_len=max_len,
        dropout=0.1
    )
    
    # 打印模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ 模型总参数量: {total_params / 1e6:.2f}M")
    
    # 训练模型
    print("\n=== 开始训练 ===")
    model = train_model(
        model, train_loader, dev_loader,
        dataset['src_vocab'], dataset['tgt_vocab'],
        epochs=epochs, lr=lr, device=device,
        model_save_path=model_save_path
    )
    
    # 测试集评估
    print("\n=== 测试集评估 ===")
    criterion = nn.CrossEntropyLoss(ignore_index=dataset['src_vocab'].token2idx[dataset['src_vocab'].pad_token]).to(device)
    # 加载最佳模型
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    else:
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    print(f"✅ 已加载最佳模型：{model_save_path}")
    test_loss, test_bleu = evaluate(model, test_loader, dataset['src_vocab'], dataset['tgt_vocab'], criterion, device)
    print(f"测试损失: {test_loss:.4f} | 测试BLEU分数: {test_bleu:.4f}")
    
    # 测试集第一句翻译验证
    print("\n=== 测试集第一句翻译验证 ===")
    test_src_sentence = dataset['test'].src_sentences[0]
    test_tgt_reference = dataset['test'].tgt_sentences[0]
    
    translated = translate_beam_search(
        model=model,
        sentence=test_src_sentence,
        src_vocab=dataset['src_vocab'],
        tgt_vocab=dataset['tgt_vocab'],
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_len=max_len,
        device=device,
        beam_size=5,
        repeat_penalty=1.5,
        temperature=0.8
    )
    
    print(f"\n📊 最终翻译对比：")
    print(f"测试集原文: {test_src_sentence}")
    print(f"模型翻译: {translated}")
    print(f"参考译文: {test_tgt_reference}")
    print("-" * 80)
    
    # 示例句子翻译
    print("\n=== 示例句子翻译 ===")
    sample_srcs = [
        "Climate change is a serious global problem.",
        "Technology can help solve many challenges.",
        "We need to protect our environment for future generations."
    ]
    for src in sample_srcs:
        print(f"\n📌 输入：{src}")
        translated = translate_beam_search(
            model=model, sentence=src, 
            src_vocab=dataset['src_vocab'], tgt_vocab=dataset['tgt_vocab'],
            device=device, beam_size=5, repeat_penalty=1.5, temperature=0.8
        )
        print(f"📌 输出：{translated}")
        print("-" * 80)


if __name__ == "__main__":
    main()