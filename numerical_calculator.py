#!/usr/bin/env python3
"""
数值计算模块 - 核心计算功能
计算置信度、自确定性等针对token的数值指标

流程：原始数据 -> 数值计算 -> 输出结构化数据
输出结构：包含pruning选中的traces的各种指标
"""

import json
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import tqdm
import re
import tempfile
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from evaluation.eval_utils_padding import extract_answer_from_output
from extension_calculator import SemanticUncertaintyCalculator

# 导入统一的答案检查函数
try:
    from voting import check_answer_equality
except ImportError:
    # 如果无法导入，定义本地版本
    def check_answer_equality(pred: str, correct: Union[str, int]) -> bool:
        """检查预测答案和正确答案是否等价"""
        if not pred or correct is None:
            return False
        
        # 将correct_answer转换为整数（如果是字符串形式的数字）
        if isinstance(correct, str) and correct.isdigit():
            correct = int(correct)
        
        # 如果correct_answer是数字，将字母转换为对应数字
        if isinstance(correct, int):
            if pred in 'ABCD':
                return ord(pred) - ord('A') == correct
            return False
        
        # 如果correct_answer是字母，直接比较
        return pred == correct


class NumericalCalculator:
    """
    数值计算器 - 计算核心评估指标
    
    主要功能：
    1. 置信度计算 (Confidence)
    2. 自确定性计算 (Self-Certainty)
    3. 样本一致性计算 (Sample Consistency)
    4. 关键词权重调整系统
    5. 两阶段处理流程
    """
    
    def __init__(self, 
                 model_name: str = None,
                 device: str = "auto",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 num_samples: int = 5,
                 temperature: float = 0.7,
                 first_stage_token_limit: int = 100):
        """
        初始化数值计算器
        
        Args:
            model_name: 模型名称
            device: 计算设备
            embedding_model: 嵌入模型名称
            num_samples: 多样本数量
            temperature: 采样温度
            first_stage_token_limit: 第一阶段计算的前N个token数量
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = embedding_model
        self.num_samples = num_samples
        self.temperature = temperature
        self.first_stage_token_limit = first_stage_token_limit
        self.tokenizer_path = None
        
        # 初始化模型（如果需要）
        self.model = None
        self.tokenizer = None
        self.embedding_model_obj = None
        
        if model_name:
            self._load_model()
        
        # 初始化嵌入模型
        self._load_embedding_model()
        
        # 初始化语义不确定性计算器
        self.semantic_calculator = SemanticUncertaintyCalculator(device=self.device)

    def _load_tokenizer_only(self, tokenizer_path: str):
        """仅加载分词器，不加载模型。"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                padding=True,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.tokenizer.padding_side = "right"
        except Exception as e:
            print(f"⚠️  无法加载tokenizer {tokenizer_path}: {e}")
            self.tokenizer = None
    
    def confidence_logprob_sum(self, logprob_sum: torch.Tensor, attention_mask: torch.Tensor, V: int):
        """
        计算self-certainty分数 - 基于Self-Certainty论文的实现
        
        参考文档：Self-Certainty论文中的公式
        公式：-1/(nV) * Σ log(V * p(j|x, y<i))
        
        Args:
            logprob_sum: torch.Tensor, shape (batch_size, seq_length) or (seq_length)
            attention_mask: torch.Tensor, shape (batch_size, seq_length) or (seq_length)  
            V: int, the vocab size
            
        Returns:
            batch_confidence_list: List[float], self-certainty分数列表
        """
        logprob_sum = logprob_sum.contiguous()
        attention_mask = attention_mask.contiguous()
        V_tensor = torch.tensor(V, dtype=logprob_sum.dtype, device=logprob_sum.device)
        conf = -1/V * logprob_sum - torch.log(V_tensor)
        valid_conf = conf * attention_mask
        batch_confidence_list = (valid_conf.sum(dim=-1) / attention_mask.sum(dim=-1)).tolist()
        return batch_confidence_list
    
    def _get_group_batch_sizes(self, base_batch_size: int) -> Dict[str, int]:
        """
        根据文本长度组获取相应的批量大小
        
        Args:
            base_batch_size: 基础批量大小
            
        Returns:
            各组对应的批量大小
        """
        return {
            "small": base_batch_size,
            "medium": max(1, base_batch_size // 2),
            "large": max(1, base_batch_size // 4)
        }
    
    def extract_code(self, output_text: str) -> str:
        """
        从输出文本中提取Python代码片段
        
        参考Self-Certainty的代码提取实现
        
        Args:
            output_text: 包含代码的输出文本
            
        Returns:
            提取的Python代码字符串
        """
        # 正则表达式模式匹配```python代码块
        pattern = re.compile(r'```python\s*\n(.*?)\n```', re.DOTALL)
        matches = pattern.findall(output_text)
        
        if not matches:
            return ""
        
        # 去除每个代码片段的空白字符
        stripped_matches = [match.strip() for match in matches]
        
        # 连接所有代码片段
        concatenated_code = "\n\n".join(stripped_matches)
        
        return concatenated_code
    
    def sanitize_math_answers(self, answer: str) -> str:
        """
        标准化数学答案格式
        
        参考Self-Certainty的数学答案处理逻辑
        
        Args:
            answer: 原始答案字符串
            
        Returns:
            标准化后的答案字符串
        """
        answer = str(answer).strip()
        # 忽略符号如$
        answer = answer.replace("$", "").strip()
        # 移除数字中的逗号
        answer = answer.replace(",", "")
        
        # Unicode到LaTeX的转换
        unicode_to_latex = {
            'π': '\\pi', '√': '\\sqrt', '∞': '\\infty', '≤': '\\leq',
            '≥': '\\geq', '≠': '\\neq', '→': '\\to', '←': '\\leftarrow',
            '∑': '\\sum', '∫': '\\int', '∂': '\\partial', '∧': '\\land',
            '≈': '\\approx', '∈': '\\in', '⊂': '\\subset', '⊆': '\\subseteq'
        }
        
        for unicode_char, latex_char in unicode_to_latex.items():
            answer = answer.replace(unicode_char, latex_char)
        
        # 转换为浮点数然后转换为整数（如果是整数）
        try:
            num = float(answer)
            if num.is_integer():
                answer = str(int(num))
            else:
                answer = str(num)
        except:
            pass
        
        # 移除空格
        answer = answer.replace(" ", "")
        return answer
    
    def convert_AAAAA_to_A(self, answer: str) -> str:
        """
        将AAAAA格式转换为A格式
        
        Args:
            answer: 原始答案
            
        Returns:
            转换后的答案
        """
        answer = str(answer).strip()
        if len(answer) >= 4 and answer[0] == answer[1] == answer[2] == answer[3]:
            return answer[0]
        return answer
    
    def perform_borda_voting(self, outputs: List[str], confidence_scores: List[float], 
                           power: float = 0.5, model_name: str = "Llama3.1", 
                           dataset: str = "math") -> Tuple[str, float]:
        """
        执行Borda投票机制
        
        参考Self-Certainty的Borda投票实现
        
        Args:
            outputs: 输出文本列表
            confidence_scores: 对应的置信度分数
            power: 投票权重指数
            model_name: 模型名称
            dataset: 数据集类型
            
        Returns:
            (最佳答案, 最高投票分数)
        """
        if not outputs or not confidence_scores:
            return "", 0.0
        
        # 按置信度排序获取排名
        sorted_indices = sorted(range(len(confidence_scores)), 
                              key=lambda k: confidence_scores[k], reverse=True)
        
        # 计算Borda投票分数
        votes_per_output = [len(confidence_scores) - rank for rank in range(len(confidence_scores))]
        
        # 应用幂函数权重
        votes_per_output = [vote**power for vote in votes_per_output]
        
        # 创建投票映射
        votes_map = {sorted_indices[i]: votes_per_output[i] for i in range(len(sorted_indices))}
        votes = [0 for _ in range(len(confidence_scores))]
        
        # 执行投票逻辑
        for i in range(len(confidence_scores)):
            try:
                answer_i = extract_answer_from_output(outputs[i], model_name, dataset)
                if answer_i is None:
                    continue
                    
                find_answer = False
                for j in range(i):
                    try:
                        answer_j = extract_answer_from_output(outputs[j], model_name, dataset)
                        if answer_j is None:
                            continue
                        if answer_i == answer_j:
                            votes[j] += votes_map[i]
                            find_answer = True
                            break
                    except:
                        continue
                        
                if not find_answer:
                    votes[i] += votes_map[i]
                    
            except:
                continue
        
        # 找到最高投票分数的答案
        best_vote_score = max(votes)
        best_index = votes.index(best_vote_score)
        
        return outputs[best_index], best_vote_score
    
    
    
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        截取文本的前N个token
        
        Args:
            text: 原始文本
            max_tokens: 最大token数量
            
        Returns:
            截取后的文本
        """
        if not self.tokenizer:
            # 如果没有tokenizer，简单按字符截取
            return text[:max_tokens * 4]  # 粗略估计：1个token约等于4个字符
        
        try:
            # 使用tokenizer进行精确截取
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= max_tokens:
                return text
            
            # 截取前max_tokens个token并解码
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return truncated_text
        except Exception as e:
            print(f"Warning: Error truncating text: {e}")
            # 回退到字符截取
            return text[:max_tokens * 4]
    
    def _count_tokens(self, text: str) -> int:
        """
        统计单段文本的token数量。
        优先使用tokenizer精确统计；若不可用，则按字符数近似估计（1 token ≈ 4 chars）。
        """
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return int(len(tokens))
            except Exception:
                pass
        # 近似回退
        return max(1, len(text) // 4)

    def count_total_tokens(self, all_outputs: List[str], selected_outputs: List[str], first_stage_token_limit: int) -> int:
        """
        计算本题总token使用量。
        公式：((总trace数 - 入围trace数M) * 第一轮读取token数N) + 入围trace全长之和。
        注意：不对入围trace重复计入第一轮的N。
        """
        total_traces = len(all_outputs)
        m_selected = len(selected_outputs)
        non_selected = max(0, total_traces - m_selected)
        stage1_tokens = non_selected * int(first_stage_token_limit)
        full_selected_tokens = 0
        for output in selected_outputs:
            full_selected_tokens += self._count_tokens(output)
        return int(stage1_tokens + full_selected_tokens)
    
    def _load_model(self):
        """
        加载语言模型 - 优化版本
        
        参考Self-Certainty的模型加载优化：
        - 添加padding token处理
        - 优化内存使用
        - 支持断点续传
        """
        try:
            torch.set_grad_enabled(False)
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                padding=True,
                trust_remote_code=True
            )
            
            # 添加padding token如果不存在
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            
            # 设置padding方向
            self.tokenizer.padding_side = "right"
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            ).to(self.device)
            
            # 更新模型配置以支持padding token
            if self.tokenizer.pad_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                # 调整embedding层大小
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # 设置为评估模式
            self.model.eval()
            
        except Exception as e:
            print(f"⚠️  无法加载模型 {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
    
    def _load_embedding_model(self):
        """加载嵌入模型"""
        try:
            self.embedding_model_obj = SentenceTransformer(self.embedding_model)
        except Exception as e:
            print(f"⚠️  无法加载embedding模型 {self.embedding_model}: {e}")
            self.embedding_model_obj = None
    
    def _load_examples(self, filepath: str, output_field_name: str = "output") -> List[Dict]:
        """
        加载示例数据 - 支持JSON和Parquet格式
        
        参考Self-Certainty的文件处理实现
        
        Args:
            filepath: 文件路径
            output_field_name: 输出字段名称
            
        Returns:
            数据列表
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".json":
            with open(filepath, "r", encoding='utf-8') as f:
                return json.load(f)
        elif ext == ".parquet":
            df = pd.read_parquet(filepath)
            # 如果output列是JSON编码的字符串，解码它
            if df[output_field_name].dtype == object and isinstance(df.iloc[0][output_field_name], str):
                df[output_field_name] = df[output_field_name].apply(json.loads)
            return df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported input format: {ext}")
    
    def _save_with_checkpoint(self, data: List[Dict], output_file: str) -> bool:
        """
        带检查点的保存功能
        
        参考Self-Certainty的断点续传实现
        
        Args:
            data: 要保存的数据
            output_file: 输出文件路径
            
        Returns:
            是否保存成功
        """
        try:
            # 使用临时文件确保原子性写入
            with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(output_file)) as tmp_file:
                json.dump(
                    data,
                    tmp_file,
                    indent=4,
                    default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o,
                    ensure_ascii=False
                )
                temp_name = tmp_file.name
            
            # 原子性替换
            os.replace(temp_name, output_file)
            return True
            
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def _load_checkpoint(self, output_file: str) -> List[Dict]:
        """
        加载检查点数据
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            已处理的数据列表
        """
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Loaded {len(data)} already processed items from checkpoint.")
                    return data
            except json.JSONDecodeError:
                print("Checkpoint file is corrupted or empty. Starting fresh.")
                return []
        return []
    
    def _load_sensitive_words(self) -> List[str]:
        """
        加载敏感词库
        
        Returns:
            敏感词列表
        """
        try:
            with open('sensitive_words.json', 'r', encoding='utf-8') as f:
                sensitive_words = json.load(f)
            return sensitive_words
        except Exception as e:
            print(f"Warning: Could not load sensitive words: {e}")
            return []
    
    def _detect_keywords_and_get_weights(self, text: str, sensitive_words: List[str], 
                                       window_size: int = 50, weight_multiplier: float = 10.0) -> List[float]:
        """
        检测关键词并计算权重
        
        Args:
            text: 输入文本
            sensitive_words: 敏感词列表
            window_size: 关键词附近窗口大小（token数）
            weight_multiplier: 权重倍数
            
        Returns:
            每个token位置的权重列表
        """
        if not sensitive_words:
            return [1.0] * len(text.split())  # 默认权重为1
        
        # 将文本分割为tokens（简单按空格分割）
        tokens = text.split()
        weights = [1.0] * len(tokens)
        
        # 转换为小写进行匹配
        text_lower = text.lower()
        
        for keyword in sensitive_words:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                # 找到关键词在文本中的位置
                start_pos = text_lower.find(keyword_lower)
                if start_pos != -1:
                    # 计算关键词在token列表中的位置
                    text_before_keyword = text[:start_pos]
                    token_pos = len(text_before_keyword.split())
                    
                    # 计算窗口范围
                    window_start = max(0, token_pos - window_size // 2)
                    window_end = min(len(tokens), token_pos + len(keyword.split()) + window_size // 2)
                    
                    # 应用权重
                    for i in range(window_start, window_end):
                        if i < len(weights):
                            weights[i] *= weight_multiplier
        
        return weights

    def _detect_keyword_weights_tokenized(self, text: str, sensitive_words: List[str], tokenizer: Any,
                                          window_size_tokens: int = 50, weight_multiplier: float = 10.0) -> List[float]:
        """
        使用tokenizer的offset_mapping将关键词命中从字符区间精确映射到token索引，返回与input_ids长度一致的权重。

        Args:
            text: 原始文本
            sensitive_words: 关键词列表
            tokenizer: 分词器（需支持return_offsets_mapping）
            window_size_tokens: 以token为单位的窗口大小
            weight_multiplier: 命中窗口内的权重倍数

        Returns:
            与token序列长度一致的权重列表
        """
        try:
            enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        except Exception:
            # 回退到旧的基于空格的实现（不精确）
            return self._detect_keywords_and_get_weights(text, sensitive_words, window_size_tokens, weight_multiplier)

        offsets = enc.get("offset_mapping", None)
        seq_len = len(enc.get("input_ids", []))
        if offsets is None:
            return [1.0] * seq_len

        weights = [1.0] * seq_len
        if not sensitive_words:
            return weights

        # 收集所有关键词命中的字符区间（大小写不敏感），允许多次命中
        patterns = []
        for kw in sensitive_words:
            try:
                patterns.append(re.compile(re.escape(kw), flags=re.IGNORECASE))
            except Exception:
                continue

        hit_char_spans: List[Tuple[int, int]] = []
        for pat in patterns:
            for m in pat.finditer(text):
                s, e = m.span()
                if s < e:
                    hit_char_spans.append((s, e))

        if not hit_char_spans:
            return weights

        # 将字符区间映射到token索引：凡与任一命中区间相交即计入
        hit_token_idxs = set()
        for ti, (ts, te) in enumerate(offsets):
            if ts == te:
                continue
            for (cs, ce) in hit_char_spans:
                if not (te <= cs or ts >= ce):
                    hit_token_idxs.add(ti)
                    break

        if not hit_token_idxs:
            return weights

        half = max(0, window_size_tokens // 2)
        for idx in hit_token_idxs:
            s = max(0, idx - half)
            e = min(seq_len, idx + 1 + half)
            for i in range(s, e):
                weights[i] *= weight_multiplier

        return weights
 
    def calculate_confidence_scores(self, outputs: List[str], use_truncated: bool = False, 
                                  keyword_weight_multiplier: float = 10.0,
                                  keyword_window_size: int = 50) -> List[float]:
        """
        计算置信度分数 - 基于Response Probability方法（支持关键词权重调整）
        
        参考文档：CISC框架中的Response Probability方法
        公式：p_θ(r,a) = [Π^n_{i=1} p_θ(x_i|x_1...x_{i-1}, q)]^{1/n}
        
        新增功能：检测敏感词库中的关键词，对附近段落应用权重调整
        
        Args:
            outputs: 模型输出列表
            use_truncated: 是否使用截取的文本（第一阶段）
            keyword_weight_multiplier: 关键词权重倍数（默认10.0）
            keyword_window_size: 关键词附近窗口大小（token数，默认50）
            
        Returns:
            置信度分数列表
        """
        if not self.model or not self.tokenizer:
            # 如果没有模型，返回模拟置信度
            return self._generate_mock_confidence_scores(outputs)
        
        # 加载敏感词库
        sensitive_words = self._load_sensitive_words()
        
        confidence_scores = []
        
        for output in outputs:
            try:
                # 根据阶段决定是否截取文本（基于真实token数量）
                if use_truncated:
                    output = self._truncate_to_tokens(output, self.first_stage_token_limit)

                # 编码输入：不添加特殊符号，保证与token级对齐一致
                enc = self.tokenizer(output, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
                enc = {k: v.to(self.device) for k, v in enc.items()}

                # 计算与真实token序列等长的关键词权重（基于offset mapping）
                token_weights = self._detect_keyword_weights_tokenized(
                    output, sensitive_words, self.tokenizer,
                    window_size_tokens=keyword_window_size, weight_multiplier=keyword_weight_multiplier
                )

                # 前向传播
                with torch.no_grad():
                    out = self.model(**enc)
                    logits = out.logits  # (1, L, V)

                input_ids = enc['input_ids']  # (1, L)
                seq_len = input_ids.shape[1]
                if seq_len < 2:
                    confidence_scores.append(0.5)
                    continue

                # Teacher forcing 对齐：用位置 i-1 的logits评估第 i 个token
                logits_shifted = logits[:, :-1, :]            # (1, L-1, V)
                labels_shifted = input_ids[:, 1:]             # (1, L-1)
                probs = F.softmax(logits_shifted, dim=-1)
                tok_probs = probs.gather(dim=-1, index=labels_shifted.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

                # 对齐权重到被评估的labels（丢弃权重[0]）
                w = torch.tensor(token_weights[1:1+tok_probs.shape[1]], dtype=tok_probs.dtype, device=tok_probs.device)
                if w.numel() == 0:
                    confidence = 0.5
                else:
                    logp = torch.log(torch.clamp(tok_probs, min=1e-8)).squeeze(0)  # (L-1,)
                    confidence = torch.exp((w * logp).sum() / torch.clamp(w.sum(), min=1e-8)).item()

                confidence_scores.append(float(confidence))

            except Exception as e:
                print(f"Error calculating confidence: {e}")
                confidence_scores.append(0.5)  # 默认值
        
        return confidence_scores
    
    def calculate_self_certainty(self, outputs: List[str], use_truncated: bool = False, keyword_weight_multiplier: float = 10.0, keyword_window_size: int = 50) -> List[float]:
        """
        计算自确定性 (Self-Certainty) 分数 - 基于Self-Certainty论文的实现（支持关键词权重调整）
        
        参考文档：Self-Certainty论文中的公式
        公式：-1/(nV) * Σ log(V * p(j|x, y<i))
        
        新增功能：检测敏感词库中的关键词，对附近段落应用权重调整
        
        Args:
            outputs: 模型输出列表
            use_truncated: 是否使用截取的文本（第一阶段）
            keyword_weight_multiplier: 关键词权重倍数（默认10.0）
            keyword_window_size: 关键词附近窗口大小（token数，默认50）
            
        Returns:
            自确定性分数列表
        """
        if not self.model or not self.tokenizer:
            # 如果没有模型，返回模拟分数
            return self._generate_mock_self_certainty_scores(outputs)
        
        sc_scores = []
        vocab_size = len(self.tokenizer.get_vocab())
        
        # 加载敏感词库
        sensitive_words = self._load_sensitive_words()
        
        for output in outputs:
            try:
                # 根据阶段决定是否截取文本
                if use_truncated:
                    output = self._truncate_to_tokens(output, self.first_stage_token_limit)
                
                # 编码输入：不添加特殊符号，保证与token级对齐一致
                enc = self.tokenizer(output, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                
                # 前向传播
                with torch.no_grad():
                    out = self.model(**enc)
                    logits = out.logits
                    
                # 计算self-certainty：log_softmax后在词表维度求和
                log_probs = F.log_softmax(logits, dim=-1)          # (1, L, V)
                logprob_sum = log_probs.sum(dim=-1)                 # (1, L)
                
                # 基于真实token的关键词权重，并与序列长度对齐
                token_weights = self._detect_keyword_weights_tokenized(
                    output, sensitive_words, self.tokenizer,
                    window_size_tokens=keyword_window_size, weight_multiplier=keyword_weight_multiplier
                )
                seq_length = logprob_sum.shape[1]
                weights_tensor = torch.tensor(token_weights[:seq_length], dtype=logprob_sum.dtype, device=self.device)
                if weights_tensor.numel() < seq_length:
                    padding = torch.ones(seq_length - weights_tensor.numel(), dtype=weights_tensor.dtype, device=self.device)
                    weights_tensor = torch.cat([weights_tensor, padding])
                
                weighted_logprob_sum = logprob_sum * weights_tensor.unsqueeze(0)
                
                # attention_mask：无pad时用全1
                attention_mask = enc.get('attention_mask', torch.ones_like(enc['input_ids']))
                sc_score_list = self.confidence_logprob_sum(weighted_logprob_sum, attention_mask, vocab_size)
                
                if sc_score_list:
                    sc_scores.append(float(sc_score_list[0]))
                else:
                    sc_scores.append(0.0)
                
            except Exception as e:
                print(f"Error calculating self-certainty: {e}")
                sc_scores.append(0.0)  # 默认值
        
        return sc_scores
    
    
    def calculate_sample_consistency(self, outputs: List[str], use_truncated: bool = False) -> List[float]:
        """
        计算样本一致性 (Sample Consistency) - 基于SSC的语义一致性
        
        参考文档：SSC框架中的语义一致性计算
        使用Centroid Proximity Weighting (CPW)方法
        
        Args:
            outputs: 模型输出列表
            use_truncated: 是否使用截取的文本（第一阶段）
            
        Returns:
            样本一致性分数列表
        """
        if not self.embedding_model_obj:
            # 如果没有嵌入模型，返回模拟分数
            return self._generate_mock_consistency_scores(outputs)
        
        consistency_scores = []
        
        for output in outputs:
            try:
                # 根据阶段决定是否截取文本
                if use_truncated:
                    output = self._truncate_to_tokens(output, self.first_stage_token_limit)
                
                # 生成多样本（简化处理）
                samples = [output] * self.num_samples
                
                # 计算嵌入
                embeddings = self.embedding_model_obj.encode(samples)
                
                # 使用Centroid Proximity Weighting (CPW)方法
                # 计算所有嵌入向量的质心
                centroid = np.mean(embeddings, axis=0)
                
                # 计算每个向量到质心的距离
                distances = []
                for embedding in embeddings:
                    distance = np.linalg.norm(embedding - centroid)
                    distances.append(distance)
                
                # 计算平均距离（距离越小，一致性越高）
                avg_distance = np.mean(distances)
                
                # 将距离转换为一致性分数（距离越小，分数越高）
                # 使用指数衰减函数：consistency = exp(-distance)
                consistency_score = np.exp(-avg_distance)
                
                consistency_scores.append(consistency_score)
                
            except Exception as e:
                print(f"Error calculating sample consistency: {e}")
                consistency_scores.append(1.0)  # 默认值
        
        return consistency_scores
    
    def extract_answers(self, outputs: List[str], model_dir: str, dataset: str = "mmlu") -> List[str]:
        """
        从大模型输出中提取推理得到的答案
        
        Args:
            outputs: 大模型输出列表
            model_dir: 模型目录
            dataset: 数据集类型
            
        Returns:
            提取的大模型推理答案列表
        """
        answers = []
        for output in outputs:
            try:
                answer = extract_answer_from_output(output, model_dir, dataset)
                if answer is not None:
                    answers.append(answer)
                else:
                    answers.append("")  # 空答案
            except Exception as e:
                print(f"Error extracting answer: {e}")
                answers.append("")
        return answers
    
    def get_correct_answer(self, item: Dict[str, Any]) -> str:
        """
        获取题目的正确答案
        
        逻辑：如果correct_answer为null，那么answer就是正确答案；
             如果correct_answer不为空，那么correct_answer才是正确答案
        
        Args:
            item: 数据项字典
            
        Returns:
            正确答案字符串
        """
        correct_answer = item.get("correct_answer")
        
        # 检查correct_answer是否为null、None或空字符串
        if correct_answer is None or correct_answer == "" or correct_answer == "null":
            # 如果correct_answer为空，使用answer字段作为正确答案
            return item.get("answer", "")
        else:
            # 如果correct_answer不为空，使用correct_answer作为正确答案
            return str(correct_answer)
    
    def _generate_mock_confidence_scores(self, outputs: List[str]) -> List[float]:
        """生成模拟置信度分数"""
        scores = []
        for output in outputs:
            # 基于输出长度和内容生成模拟置信度
            length_score = min(len(output) / 1000.0, 1.0)
            import random
            random_factor = random.uniform(0.8, 1.2)
            confidence = length_score * random_factor
            scores.append(confidence)
        return scores
    
    def _generate_mock_self_certainty_scores(self, outputs: List[str]) -> List[float]:
        """生成模拟自确定性分数"""
        scores = []
        for output in outputs:
            import random
            # 生成合理的SC分数范围
            sc_score = random.uniform(-2.0, 0.0)
            scores.append(sc_score)
        return scores
    
    
    def _generate_mock_consistency_scores(self, outputs: List[str]) -> List[float]:
        """生成模拟一致性分数"""
        scores = []
        for output in outputs:
            import random
            # 生成合理的一致性分数
            consistency_score = random.uniform(0.7, 1.0)
            scores.append(consistency_score)
        return scores
    
    def process_data(self, input_file: str, output_file: str = None, save_interval: int = 1, max_items: int = None, k: int = 32, keyword_weight_multiplier: float = 10.0, keyword_window_size: int = 50) -> str:
        """
        处理数据文件，实现两阶段流程：
        1. 第一阶段：计算前N个token的指标，用于pruning筛选
        2. 第二阶段：对pruning选出的前M个trace计算完整指标
        
        Args:
            input_file: 输入JSON文件路径
            output_file: 输出文件路径
            save_interval: 每处理多少个项目保存一次（默认1，即每个都保存）
            max_items: 最大处理项目数量（默认None，处理所有项目）
            k: pruning选择的trace数量
            
        Returns:
            输出文件路径
        """
        # 开始处理（简化输出）
        
        # 加载数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 限制处理的项目数量
        if max_items is not None:
            # 上限设为1000
            if max_items > 1000:
                max_items = 1000
            
            if max_items < len(data):
                # 随机选择起始位置
                import random
                max_start_index = len(data) - max_items
                start_index = random.randint(0, max_start_index)
                data = data[start_index:start_index + max_items]
        
        # 确定输出文件路径
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            # 如果输入文件已经包含_numerical_results，则直接使用
            if "_numerical_results" in base_name:
                output_file = f"{base_name}.json"
            else:
                output_file = f"{base_name}_numerical_results.json"
        
        # 获取模型信息
        if data and "generator" in data[0]:
            model_dir = data[0]["generator"]
            dataset = "mmlu" if data[0].get("dataset") != "crux" else "crux"
        else:
            model_dir = "unknown"
            dataset = "mmlu"
        
        results = []
        
        # 初始化输出文件（创建空数组）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        
        for index, item in enumerate(tqdm.tqdm(data, desc="Processing", unit="问题")):
            # 获取输出
            all_outputs = item["output"]
            
            # ========== 第一阶段：快速筛选 ==========
            first_stage_confidence = self.calculate_confidence_scores(all_outputs, use_truncated=True, keyword_weight_multiplier=keyword_weight_multiplier, keyword_window_size=keyword_window_size)
            first_stage_self_certainty = self.calculate_self_certainty(all_outputs, use_truncated=True, keyword_weight_multiplier=keyword_weight_multiplier, keyword_window_size=keyword_window_size)
            
            # 计算第一阶段semantic uncertainty（为每个trace单独计算，使用截取文本）
            first_stage_log_likelihoods = [np.random.normal(-2.0, 0.5) for _ in all_outputs]
            first_stage_semantic_uncertainty = self.semantic_calculator.calculate_individual_semantic_uncertainty(
                all_outputs, first_stage_log_likelihoods, use_truncated=True, 
                first_stage_token_limit=self.first_stage_token_limit, tokenizer=self.tokenizer
            )

            # 扩展小指标（第一阶段，截断）- 已完全移除
            # 原计算代码已注释，用于pruning的辅助指标仍保留占位值
            # if not (self.model and self.tokenizer):
            #     print("  - Warning: model/tokenizer not loaded, perplexity and entropy_topk will be zeros. Pass --model_name to enable.")
            # first_stage_perplexity = [
            #     self.semantic_calculator.calculate_perplexity_single_trace(o, self.model, self.tokenizer, use_truncated=True, first_stage_token_limit=self.first_stage_token_limit)
            #     for o in all_outputs
            # ] if self.model and self.tokenizer else [0.0] * len(all_outputs)
            # first_stage_entropy_topk = [
            #     self.semantic_calculator.calculate_entropy_topk_single_trace(o, self.model, self.tokenizer, k=5, temperature=1.0, use_truncated=True, first_stage_token_limit=self.first_stage_token_limit)
            #     for o in all_outputs
            # ] if self.model and self.tokenizer else [{"mean_entropy": 0.0, "std_entropy": 0.0, "mean_concentration": 0.0, "last_entropy": 0.0, "last_concentration": 0.0} for _ in all_outputs]
            
            # 为pruning保留占位值（不输出到最终结果）
            first_stage_perplexity = [0.0] * len(all_outputs)
            first_stage_entropy_topk = [{"mean_entropy": 0.0, "std_entropy": 0.0, "mean_concentration": 0.0, "last_entropy": 0.0, "last_concentration": 0.0} for _ in all_outputs]
            
            # 运行pruning选择前k个trace
            try:
                from prunning import prunning
                # 使用individual semantic uncertainty的composite_uncertainty作为consistency的替代
                semantic_uncertainty_scores = [unc['composite_uncertainty'] for unc in first_stage_semantic_uncertainty]
                # 额外小指标：使用mean_entropy和perplexity作为辅助信号（若prunning支持则传入）
                aux_entropy = [m.get("mean_entropy", 0.0) for m in first_stage_entropy_topk]
                aux_perplexity = first_stage_perplexity
                try:
                    selected_indices = prunning(
                        confidence=first_stage_confidence,
                        self_certainty=first_stage_self_certainty,
                        consistency=semantic_uncertainty_scores,
                        k=k,
                        entropy=aux_entropy,
                        perplexity=aux_perplexity
                    )
                except TypeError:
                    # 兼容旧版签名
                    selected_indices = prunning(
                        confidence=first_stage_confidence,
                        self_certainty=first_stage_self_certainty,
                        consistency=semantic_uncertainty_scores,
                        k=k
                    )
                # 确保返回的是Python列表
                if hasattr(selected_indices, 'tolist'):
                    selected_indices = selected_indices.tolist()
            except Exception as e:
                # 如果pruning失败，选择前k个
                selected_indices = list(range(min(k, len(all_outputs))))

            # ========== 第二阶段：精确计算 ==========
            selected_outputs = [all_outputs[i] for i in selected_indices if i < len(all_outputs)]
            
            # 对选中的trace计算完整指标
            final_confidence = self.calculate_confidence_scores(selected_outputs, use_truncated=False, keyword_weight_multiplier=keyword_weight_multiplier, keyword_window_size=keyword_window_size)
            final_self_certainty = self.calculate_self_certainty(selected_outputs, use_truncated=False, keyword_weight_multiplier=keyword_weight_multiplier, keyword_window_size=keyword_window_size)
            
            # 计算第二阶段semantic uncertainty（为每个trace单独计算，使用完整文本）
            final_log_likelihoods = [np.random.normal(-2.0, 0.5) for _ in selected_outputs]
            final_semantic_uncertainty = self.semantic_calculator.calculate_individual_semantic_uncertainty(
                selected_outputs, final_log_likelihoods, use_truncated=False,
                first_stage_token_limit=self.first_stage_token_limit, tokenizer=self.tokenizer
            )

            # 扩展小指标（第二阶段，完整）- 已完全移除
            # 这些指标不再计算也不输出到结果中
            # 原计算代码保留在注释中供参考：
            # final_perplexity = [
            #     self.semantic_calculator.calculate_perplexity_single_trace(o, self.model, self.tokenizer, use_truncated=False, first_stage_token_limit=self.first_stage_token_limit)
            #     for o in selected_outputs
            # ] if self.model and self.tokenizer else [0.0] * len(selected_outputs)
            # final_entropy_topk = [
            #     self.semantic_calculator.calculate_entropy_topk_single_trace(o, self.model, self.tokenizer, k=5, temperature=1.0, use_truncated=False, first_stage_token_limit=self.first_stage_token_limit)
            #     for o in selected_outputs
            # ] if self.model and self.tokenizer else [{"mean_entropy": 0.0, "std_entropy": 0.0, "mean_concentration": 0.0, "last_entropy": 0.0, "last_concentration": 0.0} for _ in selected_outputs]
            
            # 提取大模型推理得到的答案（从选中的输出中提取）
            model_answers = self.extract_answers(selected_outputs, model_dir, dataset)
            
            # 获取正确答案
            correct_answer = self.get_correct_answer(item)

            # 逐题投票（按指定指标与power聚合）
            metric_name = getattr(self, "voting_metric", "confidence")
            power = float(getattr(self, "voting_power", 0.5))
            scores = {}
            def _to_letter(x):
                try:
                    if isinstance(x, int):
                        if 0 <= x <= 3:
                            return chr(ord('A') + x)
                    s = str(x).strip()
                    if s.isdigit():
                        n = int(s)
                        if 0 <= n <= 3:
                            return chr(ord('A') + n)
                    if len(s) == 1 and s in 'ABCD':
                        return s
                except:
                    pass
                return ""
            for i, ans in enumerate(model_answers):
                ans_letter = _to_letter(ans)
                if not ans_letter:
                    continue
                if metric_name == "self_certainty":
                    if i >= len(final_self_certainty):
                        continue
                    value = float(final_self_certainty[i])
                    norm = float(np.exp(value))
                else:  # 默认使用confidence
                    if i >= len(final_confidence):
                        continue
                    value = float(final_confidence[i])
                    norm = max(0.0, value)
                weight = float(norm) ** power
                scores[ans_letter] = scores.get(ans_letter, 0.0) + weight
            voted_answer = max(scores, key=scores.get) if scores else ""

            # 评估正确性（使用统一的答案检查函数）
            is_correct = check_answer_equality(voted_answer, correct_answer)
     
            # 计算token使用量：((64 - M) * N) + 入围trace的token全长
            token_count = self.count_total_tokens(all_outputs, selected_outputs, self.first_stage_token_limit)
             
            # 构建结果项（已移除perplexity和entropy_topk）
            result_item = {
                "id": item.get("session_id", item.get("id", index)),
                "question": item.get("question", ""),
                "output": selected_outputs,
                "confidence": final_confidence,
                "self_certainty": final_self_certainty,
                "semantic_uncertainty": final_semantic_uncertainty,
                # "perplexity": final_perplexity,  # 已移除
                # "entropy_topk": final_entropy_topk,  # 已移除
                "answer": model_answers,
                "correct_answer": correct_answer,
                "voted_answer": voted_answer,
                "is_correct": is_correct,
                "token_count": token_count
            }
             
            results.append(result_item)
            
            # 边运行边保存（只保存pruning选中的traces的完整计算结果）
            if (index + 1) % save_interval == 0 or index == len(data) - 1:
                try:
                    # 转换numpy类型为Python原生类型
                    converted_results = self._convert_numpy_types(results)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(converted_results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"\n⚠️  保存文件出错: {e}")
        
        # 计算总体统计
        total_correct = sum(1 for r in results if r.get("is_correct", False))
        total_questions = len(results)
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        total_tokens = sum(r.get("token_count", 0) for r in results)
        
        print(f"\n" + "="*80)
        print(f"Two-stage numerical calculation completed!")
        print(f"="*80)
        print(f"Results saved to: {output_file}")
        print(f"\n统计信息:")
        print(f"  总问题数: {total_questions}")
        print(f"  正确数: {total_correct}")
        print(f"  准确率: {accuracy:.2%} ({accuracy:.4f})")
        print(f"  总Token使用: {total_tokens:,}")
        print(f"  平均Token/题: {total_tokens/total_questions:.0f}" if total_questions > 0 else "  平均Token/题: N/A")
        print(f"\n配置:")
        print(f"  第一轮token限制: {self.first_stage_token_limit}")
        print(f"  Pruning选择trace数: {k}")
        print(f"  投票指标: {getattr(self, 'voting_metric', 'confidence')}")
        print(f"  投票power: {getattr(self, 'voting_power', 0.5)}")
        print(f"  关键词权重倍数: {keyword_weight_multiplier}")
        print(f"  关键词窗口大小: {keyword_window_size}")
        print("="*80)
        
        return output_file

    def _convert_numpy_types(self, data):
        """
        递归转换numpy类型为Python原生类型，以便JSON序列化
        
        Args:
            data: 要转换的数据
            
        Returns:
            转换后的数据
        """
        if isinstance(data, dict):
            return {key: self._convert_numpy_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def run_voting_only(self, input_file: str) -> str:
        """
        只运行voting流程（因为pruning已经在process_data中完成）
        
        Args:
            input_file: 数值计算结果文件路径（已经包含pruning选中的traces）
            
        Returns:
            最终结果文件路径
        """
        import sys
        import os
        
        # 添加当前目录到路径，以便导入voting模块
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        try:
            from voting import weighted_vote_from_json, save_results
        except ImportError as e:
            print(f"Error importing voting module: {e}")
            return None
        
        # 读取数值计算结果（已经包含pruning选中的traces）
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 选择指标并运行voting（默认使用confidence，可通过CLI指定）
        metric_name = getattr(self, "voting_metric", "confidence")
        power = getattr(self, "voting_power", 0.5)
        voting_results = weighted_vote_from_json(input_file, metric_name=metric_name, confidence_power=power)
        
        # 保存最终结果
        final_output_file = input_file.replace("_numerical_results.json", f"_final_results_{metric_name}_power_{power}.json")
        save_results(voting_results, data, metric_name, power)
        
        return final_output_file

    def run_full_pipeline(self,
                          input_file: str,
                          output_file: Optional[str] = None,
                          save_interval: int = 1,
                          max_items: Optional[int] = None,
                          k: int = 32,
                          first_stage_token_limit: int = 100,
                          voting_metric: str = "confidence",
                          voting_power: float = 0.5,
                          keyword_weight_multiplier: float = 10.0,
                          keyword_window_size: int = 50,
                          run_voting: bool = True) -> Tuple[str, Optional[str]]:
        """
        统一全流程运行：数值计算（两阶段）+ 可选的voting。
        """
        # 同步可调参数
        self.first_stage_token_limit = first_stage_token_limit
        self.voting_power = voting_power
        self.voting_metric = voting_metric

        # 数值计算
        numerical_file = self.process_data(
            input_file=input_file,
            output_file=output_file,
            save_interval=save_interval,
            max_items=max_items,
            k=k,
            keyword_weight_multiplier=keyword_weight_multiplier,
            keyword_window_size=keyword_window_size
        )

        # 可选投票
        final_file = None
        if run_voting:
            final_file = self.run_voting_only(numerical_file)

        return numerical_file, final_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数值计算模块 - 两阶段流程")
    parser.add_argument("--input_file", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    parser.add_argument("--model_name", type=str, default="openpangu-embedded-7b-model", help="模型（或本地目录）名称；默认openpangu-embedded-7b-model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="仅加载分词器的本地目录或名称（不加载模型，默认继承model_name）")
    parser.add_argument("--load_model", action="store_true", help="是否加载模型权重（默认不加载，仅加载分词器）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="嵌入模型")
    parser.add_argument("--num_samples", type=int, default=5, help="多样本数量")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--first_stage_token_limit", type=int, default=100, help="第一阶段计算的前N个token数量")
    parser.add_argument("--save_interval", type=int, default=1, help="每处理多少个项目保存一次（默认1，即每个都保存）")
    parser.add_argument("--max_items", type=int, default=None, help="最大处理项目数量（默认None，处理所有项目）")
    parser.add_argument("--run_voting", action="store_true", help="是否运行voting流程")
    parser.add_argument("--run_prunning_voting", action="store_true", help="是否运行prunning和voting流程（兼容性参数）")
    parser.add_argument("--k", type=int, default=32, help="pruning选择的trace数量")
    parser.add_argument("--keyword_weight_multiplier", type=float, default=10.0, help="关键词权重倍数（默认10.0）")
    parser.add_argument("--keyword_window_size", type=int, default=50, help="关键词窗口大小（token数，默认50）")
    parser.add_argument("--voting_metric", type=str, default="confidence", help="voting使用的指标（confidence/self_certainty/consistency/TEMP等）")
    parser.add_argument("--voting_power", type=float, default=0.5, help="voting权重指数power")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # 创建数值计算器
    calculator = NumericalCalculator(
        model_name=(args.model_name if args.load_model else None),
        device=args.device,
        embedding_model=args.embedding_model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        first_stage_token_limit=args.first_stage_token_limit
    )
    # 默认仅加载分词器：优先tokenizer_path，否则继承model_name；若--load_model已加载则跳过
    if calculator.tokenizer is None:
        tokenizer_src = args.tokenizer_path if args.tokenizer_path else args.model_name
        if tokenizer_src:
            calculator._load_tokenizer_only(tokenizer_src)
    
    # 统一全流程：数值计算 + 可选voting
    calculator.voting_metric = args.voting_metric
    calculator.voting_power = args.voting_power
    numerical_file, final_file = calculator.run_full_pipeline(
        input_file=args.input_file,
        output_file=args.output_file,
        save_interval=args.save_interval,
        max_items=args.max_items,
        k=args.k,
        first_stage_token_limit=args.first_stage_token_limit,
        voting_metric=args.voting_metric,
        voting_power=args.voting_power,
        keyword_weight_multiplier=args.keyword_weight_multiplier,
        keyword_window_size=args.keyword_window_size,
        run_voting=(args.run_voting or args.run_prunning_voting)
    )
    
    # process_data 已经打印了详细统计，这里只在有voting时打印最终汇总
    
    # 如果指定了运行voting（支持两种参数名称）
    if args.run_voting or args.run_prunning_voting:
        if final_file:
            print(f"\n" + "="*80)
            print(f"✅ Complete pipeline finished successfully!")
            print(f"="*80)
            print(f"Final output file: {final_file}")
            
            # 读取结果文件并显示最终统计
            try:
                with open(numerical_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data:
                    total_correct = sum(1 for r in data if r.get("is_correct", False))
                    total_questions = len(data)
                    accuracy = total_correct / total_questions if total_questions > 0 else 0.0
                    total_tokens = sum(r.get("token_count", 0) for r in data)
                    
                    print(f"\n📊 最终统计:")
                    print(f"  ✓ 准确率: {accuracy:.2%} ({total_correct}/{total_questions})")
                    print(f"  ✓ Token总数: {total_tokens:,}")
                    print(f"  ✓ 平均Token/题: {total_tokens/total_questions:.0f}")
            except Exception as e:
                print(f"  (无法读取统计信息: {e})")
            
            print("="*80)
        else:
            print("\n❌ Voting failed!")


if __name__ == "__main__":
    main()
