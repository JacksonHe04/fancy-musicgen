"""
MusicGen LoRA微调训练脚本
"""
import os
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import logging

from dataset import MusicGenDataset, collate_fn

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_lora_model(model, rank=12, alpha=24, dropout=0.1):
    """设置LoRA适配器"""
    # MusicGen的decoder层名称
    target_modules = [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2"
    ]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class MusicGenTrainer:
    """MusicGen训练器"""
    
    def __init__(
        self,
        model_path: str,
        train_dataset: MusicGenDataset,
        valid_dataset: MusicGenDataset,
        output_dir: str,
        lora_rank: int = 12,
        lora_alpha: int = 24,
        lora_dropout: float = 0.1,
        learning_rate: float = 3e-4,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 5,
        warmup_steps: int = 100,
        save_steps: int = 100,
        logging_steps: int = 10,
        fp16: bool = True,
    ):
        self.model_path = model_path
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载处理器和模型
        logger.info(f"Loading model from {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_path)
        # Ensure generation config values are propagated to model config for training
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.model.generation_config.pad_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.generation_config.pad_token_id
        if getattr(self.model.config, "vocab_size", None) is None:
            self.model.config.vocab_size = self.model.decoder.config.vocab_size
        
        # 设置LoRA
        logger.info("Setting up LoRA adapter")
        self.model = setup_lora_model(self.model, lora_rank, lora_alpha, lora_dropout)
        
        # 移动到GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # 训练参数
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.fp16 = fp16
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
    
    def train(self):
        """训练模型"""
        # 设置优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        # 计算总步数
        num_training_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        
        # 设置学习率调度器
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        # 训练循环
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            optimizer.zero_grad()
            
            for step, batch in enumerate(progress_bar):
                # Pad text tensors if collate_fn kept them as lists
                if isinstance(batch.get('input_ids'), list):
                    input_ids_list = [t.squeeze(0) if t.dim() > 1 else t for t in batch['input_ids']]
                    attention_mask_list = [t.squeeze(0) if t.dim() > 1 else t for t in batch['attention_mask']]
                    batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
                        input_ids_list,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    )
                    batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
                        attention_mask_list,
                        batch_first=True,
                        padding_value=0,
                    )

                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # 准备labels（音频量化tokens）
                with torch.no_grad():
                    audio_codes = self.model.audio_encoder(
                        input_values=batch['input_values'],
                        padding_mask=batch['padding_mask'],
                    ).audio_codes
                batch_size = batch['input_values'].shape[0]
                labels = audio_codes[0].reshape(batch_size * self.model.decoder.num_codebooks, -1).to(self.device)
                batch['labels'] = labels
                
                # MusicGen模型训练需要特殊处理
                # 提取input_ids（文本条件）和audio_values（音频）
                # 模型会自动处理audio encoding和decoder输入
                try:
                    # 前向传播
                    if self.fp16:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**batch)
                            # 检查是否有loss
                            if hasattr(outputs, 'loss') and outputs.loss is not None:
                                loss = outputs.loss / self.gradient_accumulation_steps
                            else:
                                # 如果没有loss，尝试计算
                                # 这通常发生在模型配置问题时
                                print(f"Warning: No loss in outputs at step {step}")
                                continue
                    else:
                        outputs = self.model(**batch)
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            loss = outputs.loss / self.gradient_accumulation_steps
                        else:
                            print(f"Warning: No loss in outputs at step {step}")
                            continue
                except Exception as e:
                    print(f"Error in forward pass at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 反向传播
                if self.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                
                # 梯度累积
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # 记录日志
                    if global_step % self.logging_steps == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        avg_loss = total_loss / (step + 1)
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}'
                        })
                        logger.info(
                            f"Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}"
                        )
                    
                    # 保存checkpoint
                    if global_step % self.save_steps == 0:
                        checkpoint_dir = self.output_dir / f"checkpoint-{global_step}"
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(str(checkpoint_dir))
                        self.processor.save_pretrained(str(checkpoint_dir))
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # 验证
            avg_train_loss = total_loss / len(self.train_loader)
            avg_valid_loss = self.validate()
            
            logger.info(
                f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, valid_loss={avg_valid_loss:.4f}"
            )
            
            # 保存最佳模型
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_model_dir = self.output_dir / "best_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(best_model_dir))
                self.processor.save_pretrained(str(best_model_dir))
                logger.info(f"Saved best model to {best_model_dir}")
        
        # 保存最终模型
        final_model_dir = self.output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(final_model_dir))
        self.processor.save_pretrained(str(final_model_dir))
        logger.info(f"Training completed. Final model saved to {final_model_dir}")
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="Validating"):
                if isinstance(batch.get('input_ids'), list):
                    input_ids_list = [t.squeeze(0) if t.dim() > 1 else t for t in batch['input_ids']]
                    attention_mask_list = [t.squeeze(0) if t.dim() > 1 else t for t in batch['attention_mask']]
                    batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
                        input_ids_list,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    )
                    batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
                        attention_mask_list,
                        batch_first=True,
                        padding_value=0,
                    )

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                audio_codes = self.model.audio_encoder(
                    input_values=batch['input_values'],
                    padding_mask=batch['padding_mask'],
                ).audio_codes
                batch_size = batch['input_values'].shape[0]
                labels = audio_codes[0].reshape(batch_size * self.model.decoder.num_codebooks, -1).to(self.device)
                batch['labels'] = labels

                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss

                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.valid_loader)
        return avg_loss


def main():
    parser = argparse.ArgumentParser(description="MusicGen LoRA微调")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/musicgen/local/musicgen-small", help="模型路径")
    parser.add_argument("--data_dir", type=str, default="./finetune/data", help="数据目录")
    parser.add_argument("--metadata_file", type=str, default="metadata.json", help="元数据文件")
    parser.add_argument("--output_dir", type=str, default="./finetune/output", help="输出目录")
    parser.add_argument("--lora_rank", type=int, default=12, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=24, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    parser.add_argument("--max_duration", type=float, default=30.0, help="最大音频时长")
    parser.add_argument("--target_duration", type=float, default=30.0, help="目标音频时长")
    
    args = parser.parse_args()
    
    # 设置HuggingFace镜像（如果需要）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # 创建数据集
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / args.metadata_file
    
    logger.info(f"Loading training dataset from {data_dir}")
    train_dataset = MusicGenDataset(
        data_dir=str(data_dir),
        metadata_path=str(metadata_path),
        processor=processor,
        max_duration=args.max_duration,
        target_duration=args.target_duration,
    )
    
    # 暂时使用相同的数据集作为验证集（实际应用中应该分开）
    valid_dataset = train_dataset
    
    # 创建训练器
    trainer = MusicGenTrainer(
        model_path=args.model_path,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()

