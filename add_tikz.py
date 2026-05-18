import re

with open('report.tex', 'r') as f:
    content = f.read()

# 1. Markov Decision Process (Section 3.1)
mdp_tikz = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=2.5cm,
    block/.style={rectangle, draw, fill=blue!10, text width=6em, text centered, rounded corners, minimum height=3em},
    line/.style={draw, -latex', thick}
]
    \node [block] (agent) {Agent};
    \node [block, below=of agent] (env) {Environment};
    
    \draw [line] (agent.east) -- +(1,0) |- node[near start, right] {Action $A_t$} (env.east);
    \draw [line] (env.west) -- +(-1,0) |- node[near start, left] {State $S_{t+1}$, Reward $R_{t+1}$} (agent.west);
\end{tikzpicture}
\caption{The standard Markov Decision Process (MDP) loop, illustrating the interaction between the agent and the environment.}
\end{figure}
"""
content = content.replace(r'\end{itemize}', r'\end{itemize}' + '\n' + mdp_tikz)

# 2. Evolution of DQN & Rainbow DQN (Section 2.2)
evolution_tikz = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=1.5cm,
    block/.style={rectangle, draw, fill=green!10, text width=8em, text centered, rounded corners, minimum height=2.5em},
    line/.style={draw, -latex', thick}
]
    \node [block] (dqn) {DQN\\(Overestimation)};
    \node [block, right=of dqn] (ddqn) {Double DQN\\(Decouples selection \& eval)};
    \node [block, right=of ddqn] (dueling) {Dueling DQN\\(Separates V and A)};
    
    \draw [line] (dqn) -- (ddqn);
    \draw [line] (ddqn) -- (dueling);
\end{tikzpicture}
\caption{The evolution from the original DQN to Double DQN and Dueling DQN, addressing action-value overestimation and variance.}
\end{figure}

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=1.0cm,
    block/.style={rectangle, draw, fill=orange!10, text width=10em, text centered, rounded corners, minimum height=2.0em}
]
    \node [block] (rainbow) {\textbf{Rainbow DQN Components}};
    \node [block, below=0.2cm of rainbow] (c1) {Double Q-learning};
    \node [block, below=0.2cm of c1] (c2) {Prioritized Replay};
    \node [block, below=0.2cm of c2] (c3) {Dueling Networks};
    \node [block, below=0.2cm of c3] (c4) {Multi-step Learning};
    \node [block, below=0.2cm of c4] (c5) {Distributional RL};
    \node [block, below=0.2cm of c5] (c6) {Noisy Nets};
\end{tikzpicture}
\caption{The integrated components of Rainbow DQN, combining six independent improvements to achieve state-of-the-art sample efficiency.}
\end{figure}
"""
content = content.replace(r'recognizing their specific utility in handling the high-variance rewards of text games.', r'recognising their specific utility in handling the high-variance rewards of text games.' + '\n' + evolution_tikz)

# 3. DistilBERT (Section 3.2.1)
distilbert_tikz = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=0.8cm,
    block/.style={rectangle, draw, fill=purple!10, text width=12em, text centered, rounded corners, minimum height=2.0em},
    line/.style={draw, -latex', thick}
]
    \node [block] (input) {Input Tokens\\([CLS] Obs [SEP] Action)};
    \node [block, below=of input] (emb) {Embedding Layer};
    \node [block, below=of emb] (t1) {Transformer Block 1-3};
    \node [block, below=of t1] (t2) {Transformer Block 4-6};
    \node [block, below=of t2] (out) {[CLS] Representation Vector};
    
    \draw [line] (input) -- (emb);
    \draw [line] (emb) -- (t1);
    \draw [line] (t1) -- (t2);
    \draw [line] (t2) -- (out);
\end{tikzpicture}
\caption{DistilBERT architecture processing the joint observation-action sequence to generate a highly contextualised embedding.}
\end{figure}
"""
content = content.replace(r'The \texttt{[CLS]} token is specifically designed to capture the global context of the entire sequence in BERT-like models. This 768-dimensional vector is then passed to the Dueling heads.', r'The \texttt{[CLS]} token is specifically designed to capture the global context of the entire sequence in BERT-like models. This 768-dimensional vector is then passed to the Dueling heads.' + '\n' + distilbert_tikz)

# 4. Partial Unfreezing (Section 3.2.3)
unfreeze_tikz = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=0.5cm,
    frozen/.style={rectangle, draw, fill=gray!30, text width=10em, text centered, minimum height=1.5em},
    unfrozen/.style={rectangle, draw, fill=green!30, text width=10em, text centered, minimum height=1.5em},
    line/.style={draw, -latex', thick}
]
    \node [frozen] (emb) {Embeddings (Blocked)};
    \node [frozen, below=of emb] (l0) {Layer 0 (Blocked)};
    \node [frozen, below=of l0] (l1) {Layer 1 (Blocked)};
    \node [frozen, below=of l1] (l2) {Layer 2 (Blocked)};
    \node [frozen, below=of l2] (l3) {Layer 3 (Blocked)};
    \node [frozen, below=of l3] (l4) {Layer 4 (Blocked)};
    \node [unfrozen, below=of l4] (l5) {Layer 5 (Unblocked)};
    \node [unfrozen, below=of l5] (head) {Task Heads (Unblocked)};
    
    \draw [line] (emb) -- (l0);
    \draw [line] (l0) -- (l1);
    \draw [line] (l1) -- (l2);
    \draw [line] (l2) -- (l3);
    \draw [line] (l3) -- (l4);
    \draw [line] (l4) -- (l5);
    \draw [line] (l5) -- (head);
\end{tikzpicture}
\caption{Visualisation of the partial unfreezing strategy: lower layers are blocked (frozen) to preserve pre-trained linguistic features, while higher layers are unblocked (fine-tuned) for task-specific adaptation.}
\end{figure}
"""
content = content.replace(r'specialise in "cooking logic."', r'specialise in "cooking logic."' + '\n' + unfreeze_tikz)

with open('report.tex', 'w') as f:
    f.write(content)
