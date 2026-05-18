import re

with open('report.tex', 'r') as f:
    content = f.read()

# 1. British English
content = content.replace('optimization', 'optimisation')
content = content.replace('Optimization', 'Optimisation')
content = content.replace('behavior', 'behaviour')
content = content.replace('analyze', 'analyse')
content = content.replace('modeling', 'modelling')
content = content.replace('generalization', 'generalisation')
content = content.replace('utilize', 'utilise')
content = content.replace('utilizing', 'utilising')
content = content.replace('emphasize', 'emphasise')
content = content.replace('analyzing', 'analysing')
content = content.replace('stabilize', 'stabilise')

# 2. Fix Bolding
content = content.replace('**interaction encoding** and **pooling strategy**', '\\textbf{interaction encoding} and \\textbf{pooling strategy}')
content = content.replace('**Partial Unfreezing**', '\\textbf{partial unfreezing}')
content = content.replace('**D3QN agent**', '\\textbf{D3QN agent}')
content = content.replace('**D3QN (Proposed)**', '\\textbf{D3QN (Proposed)}')

# 3. Fix Abstract
old_abstract = "Experimental results demonstrate that the D3QN architecture, coupled with fine-tuned representations, achieves superior performance and learning stability compared to traditional LSTM-DQN and frozen Transformer baselines."
new_abstract = "Experimental results reveal that under the current low-data training regime, the proposed D3QN architecture fails to surpass the traditional LSTM-DQN baseline, proving less effective overall. While the complex D3QN model struggles with optimisation lag, the simpler LSTM proves more adept in initial learning stages when trained on the exact same data. The findings highlight the significant stability-complexity trade-offs in modern neuro-symbolic AI and suggest that drastically increasing the training examples is necessary for the advanced architecture to yield its theoretical benefits."
content = content.replace(old_abstract, new_abstract)

# 4. Modify 4.5 Performance Analysis
old_performance_analysis = """The LSTM-DQN baseline achieved the highest mean reward in the final 10 episodes (0.081) and the most frequent non-zero rewards (34%). This suggests that in low-data regimes (50 episodes), simpler architectures with fewer parameters can converge more rapidly on initial sub-goals like item collection. The bi-directional LSTM's sequential inductive bias appears effective for parsing the relatively short observation strings in the early stages of `Cooking World'.

The Frozen Transformer-DQN showed the least consistency, with only an 18\\% frequency of non-zero rewards. This confirms that a static pre-trained representation, while rich in general linguistic features, fails to capture the task-specific semantics required for procedural reasoning in a synthetic environment.

The proposed \\textbf{D3QN agent} demonstrated competitive peak performance (0.28) but exhibited higher variance and slower convergence compared to the LSTM. Its mean reward in the final episodes (0.038) was lower than the baseline. This can be attributed to the "optimisation lag" inherent in larger Transformer architectures; the \\textbf{partial unfreezing} of Layer 5 introduces a significantly larger number of trainable parameters than the LSTM, requiring more gradient updates to stabilise. However, the D3QN's ability to reach a near-peak reward of 0.28 indicates that the Dueling architecture is capable of identifying high-value actions, even if it has not yet reached a stable policy."""

new_performance_analysis = """The results clearly show a massive problem: the proposed architecture is not working better than the baseline. In fact, it is quite less effective. Both the LSTM and the D3QN models were trained on the exact same data under identical conditions (50 episodes). The LSTM-DQN baseline achieved the highest mean reward in the final 10 episodes (0.081) and the most frequent non-zero rewards (34\\%). This establishes that in low-data regimes, simpler architectures with fewer parameters converge more rapidly on initial sub-goals like item collection. 

The Frozen Transformer-DQN showed the least consistency, with only an 18\\% frequency of non-zero rewards. This confirms that a static pre-trained representation fails to capture the task-specific semantics required.

The proposed \\textbf{D3QN agent} exhibited poor overall results. Its mean reward in the final episodes (0.038) was less than half that of the baseline, reflecting a significant failure to outperform the simpler LSTM. This underperformance is directly attributable to the severe "optimisation lag" inherent in larger Transformer architectures. The \\textbf{partial unfreezing} of Layer 5 introduces a massive number of trainable parameters compared to the LSTM. When trained on only 50 episodes, the D3QN simply does not have enough examples to adjust these parameters effectively. If we were to vastly increase the amount of training examples (e.g., to thousands of episodes), the D3QN might overcome this hurdle, but under current constraints, it remains inferior."""

content = content.replace(old_performance_analysis, new_performance_analysis)

# 5. Modify Conclusion
old_conclusion = """This research has demonstrated that the combination of Dueling Double DQN architectures and partially fine-tuned Transformer representations provides a robust framework for solving complex text-based environments. By transitioning from the navigation-focused `Coin Collector' to the procedural `Cooking World', we highlighted the limitations of standard LSTM and frozen-encoder models. The D3QN agent's superior stability and consistent sub-goal achievement validate the importance of decoupling state-value and action advantage estimation."""

new_conclusion = """This research investigated the combination of Dueling Double DQN architectures and partially fine-tuned Transformer representations for complex text-based environments. However, the experimental results conclusively show that the proposed D3QN model is currently less effective than the standard LSTM baseline. The complex architecture failed to deliver superior performance or stability within the tested 50-episode limit, despite being trained on identical data. This highlights a critical limitation: highly parameterised models like DistilBERT require significantly more training examples to overcome optimisation lag. While decoupling state-value and action advantage estimation holds theoretical promise, the practical reality in low-data regimes is that simpler, faster-converging models like the bi-directional LSTM are more reliable."""

content = content.replace(old_conclusion, new_conclusion)

# Write back
with open('report.tex', 'w') as f:
    f.write(content)
