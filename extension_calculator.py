#!/usr/bin/env python3
"""
语义不确定性计算扩展模块
包含所有与semantic uncertainty相关的计算功能

主要功能：
1. 语义熵计算 (Semantic Entropy)
2. 语义ID分组 (Semantic ID Grouping)
3. 蕴含关系判断 (Entailment Models)
4. p_ik不确定性计算
5. p_true不确定性计算
6. 综合语义不确定性度量
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from difflib import SequenceMatcher


class SemanticUncertaintyCalculator:
    """
    语义不确定性计算器
    
    主要功能：
    1. 语义熵计算
    2. 语义ID分组
    3. 蕴含关系判断
    4. p_ik和p_true不确定性计算
    5. 综合语义不确定性度量
    """
    
    def __init__(self, device: str = "auto"):
        """
        初始化语义不确定性计算器
        
        Args:
            device: 计算设备
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_semantic_ids(self, strings_list: List[str], entailment_model=None, 
                        strict_entailment: bool = False, example: Dict = None) -> List[int]:
        """
        将预测列表分组为语义含义
        
        参考Semantic Uncertainty的语义ID分组实现
        
        Args:
            strings_list: 预测字符串列表
            entailment_model: 蕴含关系判断模型
            strict_entailment: 是否使用严格的蕴含关系
            example: 示例数据（包含问题等上下文）
            
        Returns:
            语义ID列表
        """
        if entailment_model is None:
            # 如果没有蕴含模型，使用简单的字符串匹配
            semantic_set_ids = [-1] * len(strings_list)
            next_id = 0
            
            for i, string1 in enumerate(strings_list):
                if semantic_set_ids[i] == -1:
                    semantic_set_ids[i] = next_id
                    for j in range(i+1, len(strings_list)):
                        # 简单的字符串相似度判断
                        if self._are_strings_similar(string1, strings_list[j]):
                            semantic_set_ids[j] = next_id
                    next_id += 1
            
            return semantic_set_ids
        
        def are_equivalent(text1, text2):
            try:
                implication_1 = entailment_model.check_implication(text1, text2, example=example)
                implication_2 = entailment_model.check_implication(text2, text1, example=example)
                
                if strict_entailment:
                    semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
                else:
                    implications = [implication_1, implication_2]
                    # 检查是否没有矛盾(0)且不是都中性(1,1)
                    semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
                
                return semantically_equivalent
            except:
                # 如果蕴含模型出错，回退到字符串相似度
                return self._are_strings_similar(text1, text2)
        
        # 初始化所有ID为-1
        semantic_set_ids = [-1] * len(strings_list)
        next_id = 0
        
        for i, string1 in enumerate(strings_list):
            if semantic_set_ids[i] == -1:
                semantic_set_ids[i] = next_id
                for j in range(i+1, len(strings_list)):
                    if are_equivalent(string1, strings_list[j]):
                        semantic_set_ids[j] = next_id
                next_id += 1
        
        return semantic_set_ids
    
    def _are_strings_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        简单的字符串相似度判断
        
        Args:
            text1: 第一个字符串
            text2: 第二个字符串
            threshold: 相似度阈值
            
        Returns:
            是否相似
        """
        # 使用简单的编辑距离或Jaccard相似度
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity >= threshold
    
    def logsumexp_by_id(self, semantic_ids: List[int], log_likelihoods: List[float], 
                       agg: str = 'sum_normalized') -> List[float]:
        """
        对相同语义ID的概率求和
        
        参考Semantic Uncertainty的logsumexp实现
        
        Args:
            semantic_ids: 语义ID列表
            log_likelihoods: 对数似然列表
            agg: 聚合方法
            
        Returns:
            每个语义ID的对数似然
        """
        unique_ids = sorted(list(set(semantic_ids)))
        log_likelihood_per_semantic_id = []
        
        for uid in unique_ids:
            # 找到属于当前ID的位置
            id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
            # 收集这些位置的对数似然
            id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
            
            if agg == 'sum_normalized':
                # 归一化后求和
                log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
                logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
            else:
                raise ValueError(f"Unknown aggregation method: {agg}")
            
            log_likelihood_per_semantic_id.append(logsumexp_value)
        
        return log_likelihood_per_semantic_id
    
    def predictive_entropy(self, log_probs: List[float]) -> float:
        """
        计算MC估计的熵
        
        参考Semantic Uncertainty的预测熵实现
        
        Args:
            log_probs: 对数概率列表
            
        Returns:
            熵值
        """
        entropy = -np.sum(log_probs) / len(log_probs)
        return entropy
    
    def predictive_entropy_rao(self, log_probs: List[float]) -> float:
        """
        计算Rao-Blackwellized预测熵
        
        Args:
            log_probs: 对数概率列表
            
        Returns:
            熵值
        """
        entropy = -np.sum(np.exp(log_probs) * log_probs)
        return entropy
    
    def cluster_assignment_entropy(self, semantic_ids: List[int]) -> float:
        """
        从聚类分配频率估计语义不确定性
        
        参考Semantic Uncertainty的聚类分配熵实现
        
        Args:
            semantic_ids: 语义ID列表
            
        Returns:
            聚类分配熵
        """
        n_generations = len(semantic_ids)
        counts = np.bincount(semantic_ids)
        probabilities = counts / n_generations
        
        # 避免log(0)
        probabilities = probabilities[probabilities > 0]
        entropy = -(probabilities * np.log(probabilities)).sum()
        
        return entropy
    
    def calculate_semantic_entropy(self, outputs: List[str], log_likelihoods: List[float] = None,
                                 entailment_model=None, example: Dict = None) -> Dict[str, float]:
        """
        计算语义熵相关的不确定性度量
        
        参考Semantic Uncertainty的完整语义熵计算流程
        
        Args:
            outputs: 输出文本列表
            log_likelihoods: 对应的对数似然列表
            entailment_model: 蕴含关系判断模型
            example: 示例数据
            
        Returns:
            包含各种熵度量的字典
        """
        if log_likelihoods is None:
            # 如果没有提供对数似然，使用模拟值
            log_likelihoods = [np.random.normal(-2.0, 0.5) for _ in outputs]
        
        # 计算语义ID
        semantic_ids = self.get_semantic_ids(outputs, entailment_model, example=example)
        
        # 计算各种熵度量
        entropies = {}
        
        # 1. 聚类分配熵
        entropies['cluster_assignment_entropy'] = self.cluster_assignment_entropy(semantic_ids)
        
        # 2. 常规预测熵
        entropies['regular_entropy'] = self.predictive_entropy(log_likelihoods)
        
        # 3. 语义熵
        log_likelihood_per_semantic_id = self.logsumexp_by_id(semantic_ids, log_likelihoods)
        entropies['semantic_entropy'] = self.predictive_entropy_rao(log_likelihood_per_semantic_id)
        
        return entropies
    
    class BaseEntailment:
        """蕴含关系判断基类"""
        def save_prediction_cache(self):
            pass
    
    class EntailmentDeberta(BaseEntailment):
        """DeBERTa蕴含关系判断模型"""
        
        def __init__(self, device: str = "cpu"):
            self.device = device
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/deberta-v2-xlarge-mnli"
                ).to(self.device)
                self.model_loaded = True
            except Exception as e:
                print(f"Warning: Could not load DeBERTa model: {e}")
                self.model_loaded = False
        
        def check_implication(self, text1: str, text2: str, example: Dict = None) -> int:
            """
            检查text1是否蕴含text2
            
            Args:
                text1: 前提文本
                text2: 结论文本
                example: 示例数据（未使用）
                
            Returns:
                0: contradiction, 1: neutral, 2: entailment
            """
            if not self.model_loaded:
                # 回退到简单判断
                return 1 if self._simple_entailment_check(text1, text2) else 0
            
            try:
                inputs = self.tokenizer(text1, text2, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    largest_index = torch.argmax(F.softmax(logits, dim=1))
                    prediction = largest_index.cpu().item()
                
                return prediction
            except Exception as e:
                print(f"Error in DeBERTa entailment check: {e}")
                return 1
        
        def _simple_entailment_check(self, text1: str, text2: str) -> bool:
            """简单的蕴含关系检查"""
            # 使用简单的关键词匹配
            text1_words = set(text1.lower().split())
            text2_words = set(text2.lower().split())
            overlap = len(text1_words.intersection(text2_words))
            return overlap > len(text2_words) * 0.5
    
    class EntailmentLLM(BaseEntailment):
        """基于LLM的蕴含关系判断"""
        
        def __init__(self, model_name: str = "gpt-3.5-turbo"):
            self.model_name = model_name
            self.prediction_cache = {}
        
        def check_implication(self, text1: str, text2: str, example: Dict = None) -> int:
            """
            使用LLM检查蕴含关系
            
            Args:
                text1: 前提文本
                text2: 结论文本
                example: 示例数据（包含问题等上下文）
                
            Returns:
                0: contradiction, 1: neutral, 2: entailment
            """
            if example is None:
                question = "Given context"
            else:
                question = example.get('question', 'Given context')
            
            prompt = self._create_entailment_prompt(text1, text2, question)
            
            # 使用缓存
            prompt_hash = hash(prompt)
            if prompt_hash in self.prediction_cache:
                response = self.prediction_cache[prompt_hash]
            else:
                response = self._predict_with_llm(prompt)
                self.prediction_cache[prompt_hash] = response
            
            return self._parse_entailment_response(response)
        
        def _create_entailment_prompt(self, text1: str, text2: str, question: str) -> str:
            """创建蕴含关系判断的提示"""
            prompt = f"""We are evaluating answers to the question "{question}"
Here are two possible answers:
Possible Answer 1: {text1}
Possible Answer 2: {text2}
Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""
            return prompt
        
        def _predict_with_llm(self, prompt: str) -> str:
            """使用LLM进行预测"""
            # 这里可以集成OpenAI API或其他LLM服务
            # 为了演示，使用简单的规则
            return "neutral"
        
        def _parse_entailment_response(self, response: str) -> int:
            """解析LLM的响应"""
            response_lower = response.lower()[:30]
            if 'entailment' in response_lower:
                return 2
            elif 'neutral' in response_lower:
                return 1
            elif 'contradiction' in response_lower:
                return 0
            else:
                return 1  # 默认为中性
    
    def get_p_ik(self, train_embeddings: List[torch.Tensor], is_false: List[float],
                 eval_embeddings: List[torch.Tensor] = None, eval_is_false: List[float] = None) -> List[float]:
        """
        基于嵌入训练线性分类器预测模型正确性
        
        参考Semantic Uncertainty的p_ik实现
        
        Args:
            train_embeddings: 训练嵌入列表
            is_false: 训练标签（1表示错误，0表示正确）
            eval_embeddings: 评估嵌入列表
            eval_is_false: 评估标签
            
        Returns:
            评估集上的预测概率
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.model_selection import train_test_split
        except ImportError:
            print("Warning: sklearn not available, using simple heuristic for p_ik")
            return [0.5] * len(eval_embeddings) if eval_embeddings else []
        
        # 转换嵌入为numpy数组
        train_embeddings_tensor = torch.cat(train_embeddings, dim=0)
        train_embeddings_array = train_embeddings_tensor.cpu().numpy()
        
        # 分割训练和测试数据
        X_train, X_test, y_train, y_test = train_test_split(
            train_embeddings_array, is_false, test_size=0.2, random_state=42
        )
        
        # 训练逻辑回归模型
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_auroc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
        
        print(f"p_ik test accuracy: {test_accuracy:.3f}, test AUROC: {test_auroc:.3f}")
        
        # 在完整训练集上重新训练
        model_full = LogisticRegression(random_state=42)
        model_full.fit(train_embeddings_array, is_false)
        
        # 在评估集上预测
        if eval_embeddings is not None:
            eval_embeddings_tensor = torch.cat(eval_embeddings, dim=0)
            eval_embeddings_array = eval_embeddings_tensor.cpu().numpy()
            eval_predictions = model_full.predict_proba(eval_embeddings_array)
            return eval_predictions[:, 1].tolist()  # 返回错误概率
        
        return []
    
    def calculate_p_ik_uncertainty(self, outputs: List[str], embeddings: List[torch.Tensor] = None,
                                 train_embeddings: List[torch.Tensor] = None, 
                                 train_labels: List[float] = None) -> List[float]:
        """
        计算p_ik不确定性度量
        
        Args:
            outputs: 输出文本列表
            embeddings: 对应的嵌入向量
            train_embeddings: 训练嵌入向量
            train_labels: 训练标签
            
        Returns:
            p_ik不确定性分数列表
        """
        if embeddings is None:
            # 如果没有提供嵌入，使用模拟嵌入
            embeddings = [torch.randn(768) for _ in outputs]
        
        if train_embeddings is None or train_labels is None:
            # 如果没有训练数据，使用简单的启发式方法
            return [0.5] * len(outputs)
        
        # 计算p_ik
        p_ik_scores = self.get_p_ik(train_embeddings, train_labels, embeddings)
        
        return p_ik_scores
    
    def construct_few_shot_prompt(self, model, dataset, indices: List[int], prompt: str,
                                 brief: str, brief_always: bool, make_prompt_func,
                                 num_generations: int, metric_func) -> Tuple[str, Dict]:
        """
        构建p_true不确定性度量的few-shot提示
        
        参考Semantic Uncertainty的p_true实现
        
        Args:
            model: 语言模型
            dataset: 数据集
            indices: 示例索引
            prompt: 基础提示
            brief: 简要提示
            brief_always: 是否总是使用简要提示
            make_prompt_func: 提示构建函数
            num_generations: 生成数量
            metric_func: 评估指标函数
            
        Returns:
            (few-shot提示, 响应字典)
        """
        few_shot_prompt = []
        all_responses = {}
        
        for it, i in enumerate(indices):
            prompt_candidate = []
            example = dataset[i]
            question = example["question"]
            context = example.get("context", "")
            
            if it != 0:
                prompt_candidate += ['\n']
            
            prompt_candidate += ['Question: ' + question]
            prompt_candidate += ['\nBrainstormed Answers: ']
            
            current_question = make_prompt_func(context, question, None, brief, brief_always)
            local_prompt = prompt + current_question
            
            responses = []
            for j in range(num_generations + 1):
                temperature = 0.1 if j == 0 else 1.0
                
                # 模拟模型预测
                response = f"Generated response {j} for question {i}"
                responses.append(response)
                prompt_candidate += [f'{response.strip()} \n']
                
                if j == 0:
                    most_likely_response = response
                    is_correct = metric_func(response, example, model)
            
            all_responses[i] = {
                'responses': responses,
                'most_likely_response': most_likely_response,
                'is_correct': is_correct
            }
            
            prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
            prompt_candidate += ['Is the possible answer:\n']
            prompt_candidate += ['A) True\n']
            prompt_candidate += ['B) False\n']
            prompt_candidate += ['The possible answer is:']
            prompt_candidate += [' A' if is_correct else ' B']
            
            few_shot_prompt.extend(prompt_candidate)
        
        return ''.join(few_shot_prompt), all_responses
    
    def calculate_p_true(self, model, question: str, most_probable_answer: str,
                        brainstormed_answers: List[str], few_shot_prompt: str,
                        hint: bool = False) -> float:
        """
        计算p_true不确定性度量
        
        参考Semantic Uncertainty的p_true实现
        
        Args:
            model: 语言模型
            question: 问题
            most_probable_answer: 最可能的答案
            brainstormed_answers: 头脑风暴的答案列表
            few_shot_prompt: few-shot提示
            hint: 是否使用提示
            
        Returns:
            p_true的对数概率
        """
        if few_shot_prompt:
            prompt = few_shot_prompt + '\n'
        else:
            prompt = ''
        
        prompt += 'Question: ' + question
        prompt += '\nBrainstormed Answers: '
        for answer in brainstormed_answers + [most_probable_answer]:
            prompt += answer.strip() + '\n'
        prompt += 'Possible answer: ' + most_probable_answer + '\n'
        
        if not hint:
            prompt += 'Is the possible answer:\n'
            prompt += 'A) True\n'
            prompt += 'B) False\n'
            prompt += 'The possible answer is:'
        else:
            prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'
        
        # 模拟模型获取p_true
        # 在实际实现中，这里会调用模型的get_p_true方法
        log_prob = np.random.normal(-1.0, 0.5)  # 模拟对数概率
        
        return log_prob
    
    def calculate_p_true_uncertainty(self, outputs: List[str], question: str = None,
                                   few_shot_examples: List[Dict] = None) -> List[float]:
        """
        计算p_true不确定性度量
        
        Args:
            outputs: 输出文本列表
            question: 问题文本
            few_shot_examples: few-shot示例
            
        Returns:
            p_true不确定性分数列表
        """
        if question is None:
            question = "Given question"
        
        if few_shot_examples is None:
            # 使用模拟的few-shot示例
            few_shot_examples = []
        
        p_true_scores = []
        
        for output in outputs:
            # 模拟p_true计算
            # 在实际实现中，这里会使用真实的模型和few-shot示例
            p_true_score = self.calculate_p_true(
                model=None,  # 在实际实现中传入真实模型
                question=question,
                most_probable_answer=output,
                brainstormed_answers=outputs[:3],  # 使用其他输出作为头脑风暴答案
                few_shot_prompt="",  # 在实际实现中构建真实的few-shot提示
                hint=False
            )
            p_true_scores.append(p_true_score)
        
        return p_true_scores
    
    def calculate_comprehensive_semantic_uncertainty(self, outputs: List[str], 
                                                   log_likelihoods: List[float] = None,
                                                   embeddings: List[torch.Tensor] = None,
                                                   question: str = None,
                                                   entailment_model=None,
                                                   example: Dict = None) -> Dict[str, Any]:
        """
        计算综合的语义不确定性度量
        
        集成Semantic Uncertainty的所有核心功能：
        - Semantic Entropy
        - p_ik不确定性
        - p_true不确定性
        - 各种熵度量
        
        Args:
            outputs: 输出文本列表
            log_likelihoods: 对数似然列表
            embeddings: 嵌入向量列表
            question: 问题文本
            entailment_model: 蕴含关系判断模型
            example: 示例数据
            
        Returns:
            包含所有不确定性度量的字典
        """
        uncertainty_measures = {}
        
        # 1. 计算语义熵相关度量
        semantic_entropy_results = self.calculate_semantic_entropy(
            outputs, log_likelihoods, entailment_model, example
        )
        uncertainty_measures.update(semantic_entropy_results)
        
        # 2. 计算p_ik不确定性（如果有嵌入）
        if embeddings is not None:
            p_ik_scores = self.calculate_p_ik_uncertainty(outputs, embeddings)
            uncertainty_measures['p_ik'] = p_ik_scores
        
        # 3. 计算p_true不确定性
        p_true_scores = self.calculate_p_true_uncertainty(outputs, question)
        uncertainty_measures['p_true'] = p_true_scores
        
        # 4. 计算语义ID
        semantic_ids = self.get_semantic_ids(outputs, entailment_model, example=example)
        uncertainty_measures['semantic_ids'] = semantic_ids
        
        # 5. 计算聚类统计
        unique_ids = list(set(semantic_ids))
        uncertainty_measures['num_semantic_clusters'] = len(unique_ids)
        uncertainty_measures['semantic_diversity'] = len(unique_ids) / len(outputs)
        
        return uncertainty_measures
    
    def calculate_single_trace_semantic_uncertainty(self, output: str, log_likelihood: float = None, 
                                                  use_truncated: bool = False, 
                                                  first_stage_token_limit: int = 100,
                                                  tokenizer=None) -> Dict[str, float]:
        """
        计算单个trace的语义不确定性度量
        
        Args:
            output: 单个输出文本
            log_likelihood: 对数似然（如果提供）
            use_truncated: 是否使用截取的文本（第一阶段）
            first_stage_token_limit: 第一阶段token限制
            tokenizer: tokenizer对象
            
        Returns:
            单个trace的语义不确定性度量字典
        """
        if log_likelihood is None:
            log_likelihood = np.random.normal(-2.0, 0.5)
        
        # 根据阶段决定是否截取文本
        if use_truncated:
            processed_output = self._truncate_to_tokens(output, first_stage_token_limit, tokenizer)
        else:
            processed_output = output
        
        # 对于单个trace，我们计算简化的语义不确定性
        uncertainty_measures = {}
        
        # 1. 基于文本长度的简单熵估计（使用处理后的文本）
        text_length = len(processed_output.split())
        uncertainty_measures['text_length_entropy'] = np.log(max(text_length, 1))
        
        # 2. 基于对数似然的熵
        uncertainty_measures['log_likelihood_entropy'] = -log_likelihood
        
        # 3. 基于关键词密度的不确定性（使用处理后的文本）
        sensitive_words = self._load_sensitive_words()
        keyword_count = sum(1 for word in sensitive_words if word.lower() in processed_output.lower())
        uncertainty_measures['keyword_density'] = keyword_count / max(text_length, 1)
        
        # 4. 基于文本复杂度的不确定性（使用处理后的文本）
        uncertainty_measures['text_complexity'] = len(set(processed_output.lower().split())) / max(text_length, 1)
        
        # 5. 综合不确定性分数
        uncertainty_measures['composite_uncertainty'] = (
            uncertainty_measures['log_likelihood_entropy'] * 0.4 +
            uncertainty_measures['keyword_density'] * 0.3 +
            uncertainty_measures['text_complexity'] * 0.3
        )
        
        return uncertainty_measures
    
    def calculate_individual_semantic_uncertainty(self, outputs: List[str], log_likelihoods: List[float] = None, 
                                                use_truncated: bool = False,
                                                first_stage_token_limit: int = 100,
                                                tokenizer=None) -> List[Dict[str, float]]:
        """
        为每个trace单独计算语义不确定性度量
        
        Args:
            outputs: 输出文本列表
            log_likelihoods: 对应的对数似然列表
            use_truncated: 是否使用截取的文本（第一阶段）
            first_stage_token_limit: 第一阶段token限制
            tokenizer: tokenizer对象
            
        Returns:
            每个trace的语义不确定性度量列表
        """
        if log_likelihoods is None:
            log_likelihoods = [np.random.normal(-2.0, 0.5) for _ in outputs]
        
        individual_uncertainties = []
        
        for i, output in enumerate(outputs):
            log_likelihood = log_likelihoods[i] if i < len(log_likelihoods) else np.random.normal(-2.0, 0.5)
            trace_uncertainty = self.calculate_single_trace_semantic_uncertainty(
                output, log_likelihood, use_truncated, first_stage_token_limit, tokenizer
            )
            individual_uncertainties.append(trace_uncertainty)
        
        return individual_uncertainties
    
    def _truncate_to_tokens(self, text: str, max_tokens: int, tokenizer=None) -> str:
        """
        截取文本的前N个token
        
        Args:
            text: 原始文本
            max_tokens: 最大token数量
            tokenizer: tokenizer对象
            
        Returns:
            截取后的文本
        """
        if not tokenizer:
            # 如果没有tokenizer，简单按字符截取
            return text[:max_tokens * 4]  # 粗略估计：1个token约等于4个字符
        
        try:
            # 使用tokenizer进行精确截取
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= max_tokens:
                return text
            
            # 截取前max_tokens个token并解码
            truncated_tokens = tokens[:max_tokens]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return truncated_text
        except Exception as e:
            print(f"Warning: Error truncating text: {e}")
            # 回退到字符截取
            return text[:max_tokens * 4]
    
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

    # ========================= 参考文档指标实现（单trace，支持截断） =========================
    def calculate_perplexity_single_trace(self, 
                                          output: str,
                                          model,
                                          tokenizer,
                                          use_truncated: bool = False,
                                          first_stage_token_limit: int = 100) -> float:
        """
        计算单个trace的困惑度（Perplexity）。
        使用因果LM的标准方法：对输入shift计算每个目标token的log概率平均，PP = exp(-mean_log_prob)。
        """
        if tokenizer is None or model is None or not output:
            return 0.0
        text = output
        if use_truncated:
            text = self._truncate_to_tokens(output, first_stage_token_limit, tokenizer)
        try:
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(model.device if hasattr(model, 'device') else 'cpu') for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits  # [B, T, V]
                # shift
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs['input_ids'][:, 1:]
                log_probs = F.log_softmax(shift_logits, dim=-1)
                # gather target token log-probs
                gathered = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                valid_token_count = (shift_labels != tokenizer.pad_token_id).sum().item() if tokenizer.pad_token_id is not None else gathered.numel()
                if valid_token_count == 0:
                    return 0.0
                mean_log_prob = gathered.sum().item() / valid_token_count
                perplexity = float(np.exp(-mean_log_prob))
                return perplexity
        except Exception:
            return 0.0

    def calculate_entropy_topk_single_trace(self,
                                            output: str,
                                            model,
                                            tokenizer,
                                            k: int = 5,
                                            temperature: float = 1.0,
                                            use_truncated: bool = False,
                                            first_stage_token_limit: int = 100) -> Dict[str, float]:
        """
        计算单个trace的预测熵与Top-k概率集中度（按token序列求均值）。
        返回汇总统计：mean_entropy, std_entropy, mean_concentration, last_entropy, last_concentration。
        """
        results: Dict[str, float] = {}
        if tokenizer is None or model is None or not output:
            return {"mean_entropy": 0.0, "std_entropy": 0.0, "mean_concentration": 0.0, "last_entropy": 0.0, "last_concentration": 0.0}
        text = output
        if use_truncated:
            text = self._truncate_to_tokens(output, first_stage_token_limit, tokenizer)
        try:
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(model.device if hasattr(model, 'device') else 'cpu') for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits  # [B, T, V]
                logits = logits / max(1e-6, float(temperature))
                probs = F.softmax(logits, dim=-1)
                entropies = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [B, T]
                # top-k concentration per position
                topk_vals, _ = torch.topk(probs, k=min(k, probs.size(-1)))  # [B, T, k]
                concentrations = topk_vals.sum(dim=-1)  # [B, T]
                ent_list = entropies.squeeze(0).tolist()
                con_list = concentrations.squeeze(0).tolist()
                if len(ent_list) == 0:
                    return {"mean_entropy": 0.0, "std_entropy": 0.0, "mean_concentration": 0.0, "last_entropy": 0.0, "last_concentration": 0.0}
                results["mean_entropy"] = float(np.mean(ent_list))
                results["std_entropy"] = float(np.std(ent_list))
                results["mean_concentration"] = float(np.mean(con_list))
                results["last_entropy"] = float(ent_list[-1])
                results["last_concentration"] = float(con_list[-1])
                return results
        except Exception:
            return {"mean_entropy": 0.0, "std_entropy": 0.0, "mean_concentration": 0.0, "last_entropy": 0.0, "last_concentration": 0.0}

    def calculate_semantic_drift_single_trace(self,
                                               output: str,
                                               encoder_model,
                                               metric: str = 'cosine',
                                               window_size: int = 10,
                                               drift_threshold: float = 0.5,
                                               use_truncated: bool = False,
                                               first_stage_token_limit: int = 100,
                                               tokenizer=None) -> Dict[str, float]:
        """
        计算单个trace的语义漂移（基于句段embedding序列）。
        返回聚合指标：drift_from_start_mean, drift_from_start_max, local_drift_mean, is_drifting_ratio。
        """
        if encoder_model is None or not output:
            return {"drift_from_start_mean": 0.0, "drift_from_start_max": 0.0, "local_drift_mean": 0.0, "is_drifting_ratio": 0.0}
        text = output
        if use_truncated:
            text = self._truncate_to_tokens(output, first_stage_token_limit, tokenizer)
        # 简单按句号/换行切分
        import re as _re
        segments = [seg.strip() for seg in _re.split(r"[\n\r\.\!\?]+", text) if seg.strip()]
        if len(segments) == 0:
            segments = [text]
        try:
            # 获取每段embedding（假设encoder_model.encode返回numpy向量或tensor）
            embs = []
            for seg in segments:
                emb = encoder_model.encode(seg) if hasattr(encoder_model, 'encode') else None
                if emb is None:
                    continue
                if isinstance(emb, np.ndarray):
                    emb = torch.tensor(emb)
                elif not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(np.array(emb))
                embs.append(emb.float())
            if not embs:
                return {"drift_from_start_mean": 0.0, "drift_from_start_max": 0.0, "local_drift_mean": 0.0, "is_drifting_ratio": 0.0}
            initial = embs[0]
            # 计算距离
            def _dist(a: torch.Tensor, b: torch.Tensor) -> float:
                if metric == 'cosine':
                    return float(1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
                if metric == 'euclidean':
                    return float(torch.norm(a - b, p=2).item())
                if metric == 'manhattan':
                    return float(torch.norm(a - b, p=1).item())
                raise ValueError(f"Unsupported metric: {metric}")
            drift_from_start = [0.0]
            local = [0.0]
            from collections import deque
            window = deque(maxlen=max(1, int(window_size)))
            window.append(initial)
            for i in range(1, len(embs)):
                cur = embs[i]
                drift_from_start.append(_dist(cur, initial))
                if len(window) > 0:
                    wmean = torch.stack(list(window)).mean(dim=0)
                    local.append(_dist(cur, wmean))
                else:
                    local.append(0.0)
                window.append(cur)
            is_drifting = [1.0 if d > drift_threshold else 0.0 for d in drift_from_start]
            return {
                "drift_from_start_mean": float(np.mean(drift_from_start)),
                "drift_from_start_max": float(np.max(drift_from_start)),
                "local_drift_mean": float(np.mean(local)),
                "is_drifting_ratio": float(np.mean(is_drifting))
            }
        except Exception:
            return {"drift_from_start_mean": 0.0, "drift_from_start_max": 0.0, "local_drift_mean": 0.0, "is_drifting_ratio": 0.0}

    def calculate_relevance_single_trace(self,
                                         output: str,
                                         query_text: str,
                                         encoder_model,
                                         relevance_threshold: float = 0.7,
                                         aggregate_method: str = 'exponential_decay',
                                         use_truncated: bool = False,
                                         first_stage_token_limit: int = 100,
                                         tokenizer=None) -> Dict[str, float]:
        """
        计算单个trace与查询的语义相关度（按句段聚合）。
        返回：instant_relevance, aggregate_relevance, is_off_topic, relevance_trend, min_relevance, max_relevance, topic_consistency。
        """
        if encoder_model is None or not output:
            return {"instant_relevance": 0.0, "aggregate_relevance": 0.0, "is_off_topic": True, "relevance_trend": 0.0, "min_relevance": 0.0, "max_relevance": 0.0, "topic_consistency": 1.0}
        text = output
        if use_truncated:
            text = self._truncate_to_tokens(output, first_stage_token_limit, tokenizer)
        import re as _re
        segments = [seg.strip() for seg in _re.split(r"[\n\r\.\!\?]+", text) if seg.strip()]
        if len(segments) == 0:
            segments = [text]
        try:
            # embeddings
            q_emb = encoder_model.encode(query_text)
            if isinstance(q_emb, np.ndarray):
                q_emb = torch.tensor(q_emb).float()
            elif not isinstance(q_emb, torch.Tensor):
                q_emb = torch.tensor(np.array(q_emb)).float()
            rel_hist: List[float] = []
            for seg in segments:
                r_emb = encoder_model.encode(seg)
                if isinstance(r_emb, np.ndarray):
                    r_emb = torch.tensor(r_emb).float()
                elif not isinstance(r_emb, torch.Tensor):
                    r_emb = torch.tensor(np.array(r_emb)).float()
                cos_sim = F.cosine_similarity(q_emb.unsqueeze(0), r_emb.unsqueeze(0)).item()
                rel_hist.append(float(cos_sim))
            if not rel_hist:
                return {"instant_relevance": 0.0, "aggregate_relevance": 0.0, "is_off_topic": True, "relevance_trend": 0.0, "min_relevance": 0.0, "max_relevance": 0.0, "topic_consistency": 1.0}
            instant = rel_hist[-1]
            if aggregate_method == 'exponential_decay':
                weights = np.exp(-0.1 * np.arange(len(rel_hist)-1, -1, -1))
                weights = weights / weights.sum()
                aggregate = float(np.sum(np.array(rel_hist) * weights))
            elif aggregate_method == 'moving_average':
                window = min(10, len(rel_hist))
                aggregate = float(np.mean(rel_hist[-window:]))
            else:
                aggregate = float(np.mean(rel_hist))
            is_off_topic = bool(instant < relevance_threshold)
            trend = float(np.polyfit(np.arange(len(rel_hist)), rel_hist, 1)[0]) if len(rel_hist) > 1 else 0.0
            topic_consistency = float(np.exp(-2 * np.std(rel_hist))) if len(rel_hist) >= 2 else 1.0
            return {
                "instant_relevance": instant,
                "aggregate_relevance": aggregate,
                "is_off_topic": is_off_topic,
                "relevance_trend": trend,
                "min_relevance": float(min(rel_hist)),
                "max_relevance": float(max(rel_hist)),
                "topic_consistency": topic_consistency
            }
        except Exception:
            return {"instant_relevance": 0.0, "aggregate_relevance": 0.0, "is_off_topic": True, "relevance_trend": 0.0, "min_relevance": 0.0, "max_relevance": 0.0, "topic_consistency": 1.0}
