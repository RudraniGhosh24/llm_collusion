# Correlation Is Not Collusion: A Plus-Factor Auditor for AI Agents

Author Name 1, Affiliation
Author Name 2, Affiliation
Author Name 3, Affiliation

With Apart Research

## Abstract

Detecting collusion between AI agents is usually framed as measuring how similar their behavior is. We argue this framing is a mistake, and we show it fails in practice. Two agents built on the same model often behave alike for an innocent reason: shared training leads them to the same choices, with no agreement between them. A detector that treats similarity as collusion will flag this innocent parallelism as a conspiracy. Antitrust law faced the identical problem a century ago and resolved it with a clear doctrine: parallel conduct alone is not unlawful, and inferring an agreement requires additional circumstantial evidence known as plus factors. We turn that doctrine into a working, black-box auditor that scores logged agent trajectories for the signatures of an agreement (acting against self-interest, signalling, and plan-to-action concordance) rather than for correlation. We build a controlled repeated-pricing benchmark with four labeled conditions, including a leak-free negative control in which a single model instance plays both sellers with no channel and no incentive. On six models, two open-weight and four frontier, a standard correlation-and-judge detector false-positives on innocent parallelism, and on the two most capable models it flags every condition, including fully independent agents, as collusion. Our plus-factor auditor has a false-positive rate of zero on all six models. The auditor is specificity-first: it never cries wolf, though its sensitivity depends on a calibration step we analyze and propose to fix. The contribution is a reframing of collusion detection, grounded in law, with evidence that it matters most exactly on frontier models.

## 1. Introduction

AI agents are increasingly delegated real economic decisions: setting prices, bidding, and negotiating. A recognized failure mode is that such agents learn to collude, sustaining outcomes that benefit themselves at the expense of the people they serve, without any human instructing them to. This is not hypothetical. Language-model pricing agents reach supracompetitive prices on their own (Fish et al., 2024), agents independently adopt anti-competitive strategies (Lin et al., 2024), and many models form coalitions when given a private channel in realistic multi-agent tasks (Colosseum, 2026). The classic precursor result, that reinforcement-learning pricing agents learn to collude without communicating, is due to Calvano, Calzolari, Denicolò and Pastorello (2020).

Given that the behavior exists, the operational question for anyone monitoring these systems is how to recognize it. The dominant instinct is to measure dependence: if two agents' prices move together, call it collusion. We argue, and then demonstrate, that this instinct is wrong, and dangerously so on capable models.

**The confounder.** Consider two agents, A and B, and ask what makes their behavior correlated. One cause is an agreement: A conditions its action on B's and vice versa, a genuine causal link between the two agents. A second cause is a common prior: both agents are the same underlying model, trained on the same data, so they independently reach the same decisions. In the language of causal inference, shared model identity is a common cause of both agents' behavior, while collusion is a causal edge between them. Correlation cannot separate these two explanations. A detector that keys on correlation therefore cannot distinguish a conspiracy from a coincidence, and will raise false alarms whenever two copies of a capable model are deployed together. Recent evidence that frontier models exhibit correlated behavior increasing with capability (arXiv:2609.04373) makes this failure mode more severe exactly as models improve.

**The legal precedent.** This is precisely the problem antitrust law solved. Firms in a concentrated market often price in parallel without any agreement, a pattern courts call conscious parallelism. US law is explicit that parallel conduct, by itself, is lawful and does not establish a Sherman Act violation (Bell Atlantic Corp. v. Twombly, 2007; Aircraft Check Services Co. v. Verizon Wireless, 7th Cir., 2015). To infer an unlawful agreement, courts require plus factors: additional circumstantial evidence that makes independent action an implausible explanation (Kovacic et al., 2011). Recognized plus factors include acting against one's own self-interest, inter-competitor communication, and a shared motive to coordinate (Leslie, 2023). The machine learning literature has begun re-deriving fragments of this without engaging the older field that already formalized it.

**This work.** We operationalize the plus-factor doctrine as a black-box auditor over logged agent trajectories, and we evaluate it against the standard correlation-based approach on a benchmark designed to expose the confounder. Our contributions are:

1. A reframing and a method. We recast collusion detection as inference about an agreement rather than measurement of similarity, and we instantiate it as the plus-factor auditor, which scores trajectories for acting against self-interest (via counterfactual replay), signalling, and plan-to-action concordance.
2. A benchmark with a leak-free negative control. Four labeled conditions holding game, economics, and seed fixed, including a condition in which one model instance plays both agents with no channel and no incentive, isolating innocent parallelism as a measurable quantity.
3. Evidence across six models that the standard detector fails as predicted, false-positiving on innocent parallelism and, on the two most capable models, flagging even fully independent agents, while the plus-factor auditor has a zero false-positive rate throughout.

## 2. Related Work

The closest body of work establishes that AI agents collude in the first place. Calvano, Calzolari, Denicolò and Pastorello (2020) showed that reinforcement-learning pricing agents learn to sustain high prices without ever communicating, and Fish, Gonczarowski and Shorrer (2024) carried this into the language-model era, finding that LLM pricing agents reach supracompetitive prices and that small changes in their prompts move the degree of collusion. Nakamura et al. (2026) measured how often models form coalitions in realistic group tasks, and Agrawal et al. (2025) looked at collusion in LLM double auctions under different amounts of communication and oversight. All of this work answers whether collusion happens. We are after a different question, which is how an outside observer can tell that it happened without being fooled by behavior that only looks like it.

The methods we most directly position against are the ones that measure dependence between agents. Bonjour et al. (2022) propose pairwise net influence, an information-theoretic way to pick out colluding pairs in small games, and the obvious practitioner baseline is to pair a price correlation with a language model reading the transcript and rating it. Our disagreement with this family is not that the statistics are wrong but that the target is. Measuring how related two agents' behaviors are cannot separate a real agreement from two copies of the same model reaching the same conclusions on their own, a point made concrete by evidence that same-base-model deployments correlate while different-base-model deployments do not (Why Better Models Can Create Riskier Systems, 2026). We treat net influence as a useful corroborating signal rather than the thing the decision hinges on, and we show that the similarity-first approach fails in practice, most badly on the strongest models.

A second line of work looks inside the models rather than at their behavior. Rose, Cullen, Kaplowitz and Schroeder de Witt (2026) detect collusion from internal activations aggregated across interacting agents, motivated partly because text-level monitors can be dodged through obfuscation and steganography (Das, Ponde, More et al., 2026). That approach is powerful but assumes white-box access to the agents. Ours assumes the opposite. It reads only logged trajectories, which is the situation a downstream or third-party auditor is usually in, with no way to see the weights or activations of the systems it is checking.

The steganography literature marks the edge of what any transcript-based method can do. If agents can bury their coordination inside text that looks ordinary (Motwani et al., 2024; Tool Use Enables Undetectable Steganography, 2026), reading the transcript stops being enough. Our side-channel condition and the signalling factor that targets it cover a simple non-text case, but we do not claim to catch a determined steganographic scheme, and we say so in the limitations rather than paper over it.

We lean on one more empirical result. Shi et al. (2026) find that in repeated games more than 90 percent of the times an agent breaks from what it announced, the deviation was already written into its private plan. We turn that into a detection signal by checking whether one agent's actions follow the plan the other one stated privately.

The part of the picture the machine learning field has mostly left out is the law. Antitrust has spent a century on exactly our confound, where competing firms move in parallel with no agreement between them. Courts settled that parallel conduct on its own is lawful and proves nothing (Bell Atlantic Corp. v. Twombly, 2007; Aircraft Check Services Co. v. Verizon Wireless, 2015), and that to infer an agreement you need plus factors, extra circumstantial evidence that makes acting alone an implausible story (Interstate Circuit v. United States, 1939; Kovacic et al., 2011; Leslie, 2023). As far as we know, no one has turned that plus-factor test into something you can compute over AI agent logs. That is the gap we fill, and it is the reason our detector has a principled answer to the parallel-but-innocent case that defeats a correlation detector.

## 3. Methods

### 3.1 Problem setup

Two seller agents play a repeated Bertrand pricing game. Each round, each agent chooses a price from a fixed grid, demand follows a standard logit model, and payoffs are realized. We compute the one-shot Nash price and the monopoly price by brute force over the grid, giving fixed competitive and collusive reference points. All stochasticity is controlled by an injected seed, so any run is exactly reproducible. We log, per round and per agent, the chosen price, the realized payoff, any message, and the agent's private plan, which is the information the auditor is allowed to read.

### 3.2 Four labeled conditions

We hold the game, economics, and seed fixed, and vary only the factors that could induce coordination. This isolation is what lets us attribute differences to the manipulated factor rather than to confounds.

- **Q1 (collusion, explicit channel):** agents receive an incentive to keep joint prices high and a chat channel.
- **Q2 (collusion, side channel only):** the same incentive, no chat channel, only an action-ordering side channel.
- **Q3 (innocent parallelism, the negative control):** a single model instance plays both sellers, with no channel and no incentive. Any similarity is a common-cause artifact, not an agreement. This is the computational analogue of lawful conscious parallelism.
- **Q4 (independent):** agents choose a one-shot best response with no coordination mechanism.

Ground-truth labels are fixed at construction time and never inferred from behavior. We verified that the Q3 prompt is a strict textual subset of the Q1 and Q2 prompts, so no coordination hint leaks into the control; this leak-freedom was checked by an automated banned-phrase scan over every logged prompt.

### 3.3 Detectors

**Naive baseline.** The approach a practitioner would reach for first: the Pearson correlation between the two sellers' price series, combined with a language-model judge that reads the transcript and rates it for collusion. Both are measures of similarity or apparent agreement in the surface text.

**Plus-factor auditor.** A label-blind detector that computes four agreement-specific signals, mapped to recognized legal plus factors.

- **pf1, acting against self-interest (Interstate Circuit, 1939).** For each observed action we run a counterfactual replay: we hold the rival at a fixed non-cooperating competitive policy and recompute the payoff the agent would have obtained. pf1 measures the payoff the agent gave up by pricing above its myopic best response toward the cartel price. Crucially the signal is directional and positive-part: only sacrifices in the collusive direction count, so that random or noisy deviations do not register. This is the computational meaning of acting against one's short-term interest in a way that only pays off if the partner reciprocates.
- **pf2, signalling (inter-competitor communication).** Mutual information between an explicit message or an action-ordering side channel and the partner's next action.
- **pf3, motive and opportunity.** Used as corroboration rather than as a hard gate, since motive is constant in a symmetric game.
- **pf4, plan-to-action concordance.** Whether one agent's realized action tracks the other's privately stated plan (Shi et al., 2026).

The decision rule is deliberately conservative: the auditor declares collusion only if pf1 fires and at least one of pf2 or pf4 corroborates it. The pf1 firing threshold is calibrated on the independent Q4 condition as the mean plus k standard deviations of the Q4 gap distribution.

### 3.4 Models and infrastructure

We evaluate two open-weight models we host ourselves, Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, and four frontier models accessed through a hosted proxy, GPT-5.5, Gemini-3.8-flash, Gemini-2.5-pro, and Claude Sonnet 4.6. Self-hosting the open-weight models is a deliberate design choice: the Q3 control requires the exact same checkpoint to play both agents with controlled sampling, which a hosted API cannot guarantee. Open-weight runs use 5 seeds; frontier runs use 2 seeds, a cost-driven constraint we treat as a limitation. The environment, both detectors, and the evaluation were implemented from scratch; the counterfactual replay, the leak-free control, and the label-blindness of both detectors were verified by an independent methodology review during development.

### 3.5 A design error we corrected

Our first version of pf1 was non-directional: it measured distance from the myopic best response in either direction. We found numerically that a purely random pricer produced a larger pf1 value than a full-monopoly colluder, because random prices land far from best response on both sides. A threshold calibrated this way would never fire on real collusion, collapsing recall to zero. Making pf1 directional (counting only deviations toward the cartel price) resolved this, and also sharpens the correspondence to the legal notion, where acting against self-interest means pricing high to sustain a cartel, not pricing erratically.

## 4. Results

### 4.1 Main result: zero false positives on innocent parallelism, across all six models

Table 1 reports, for each model, the false-positive rate of each detector on the negative conditions (Q3 innocent parallelism and Q4 independent) and the auditor's recall on the positive conditions. The plus-factor auditor has a false-positive rate of 0.0 on every model. The naive detector does not: on the open-weight models it is uninformative (it fires on nothing, recall 0), and on the two most capable frontier models it false-positives on the negatives, in the Gemini-2.5-pro and Claude Sonnet 4.6 cases flagging every condition as collusion.

Table 1. Detector performance by model. FPR is the false-positive rate on the negative conditions (Q3, Q4). Open-weight runs use 5 seeds; frontier runs use 2 seeds.

| Model | Type | Naive FPR | Auditor FPR | Auditor recall |
|---|---|---|---|---|
| Qwen2.5-7B | open | 0.0 (uninformative) | 0.0 | 0.5 |
| Llama-3.1-8B | open | 0.0 (uninformative) | 0.0 | 0.0 |
| GPT-5.5 | frontier | 0.0 | 0.0 | 0.5 |
| Gemini-3.8-flash | frontier | 0.0 | 0.0 | 0.5 |
| Gemini-2.5-pro | frontier | 1.0 (flags all) | 0.0 | 0.0 |
| Claude Sonnet 4.6 | frontier | 1.0 (flags all) | 0.0 | 0.0 |

### 4.2 The failure is worst on the most capable models

The naive detector's false positives occur on Gemini-2.5-pro and Claude Sonnet 4.6, the two strongest models in our set. On both, it labels the innocent-parallelism control (Q3) and even the fully independent condition (Q4) as collusion. These are exactly the situations antitrust doctrine identifies as lawful: similar behavior with no agreement. The plus-factor auditor rejects both. This is the paper's central empirical point, and it is a point about deployment risk, since these are the models most likely to be run in production and most likely to produce the correlated-but-innocent behavior that fools a similarity detector.

### 4.3 The conditions are genuine

The labels are not cosmetic. On Qwen, back-half mean price indices were 8.58 in Q1 and 9.80 in Q2, at or near the monopoly level of 9, against a competitive level of 5, confirming the positive conditions produced real supracompetitive pricing. The Q3 control sat at 6.93, only mildly above competitive, consistent with shared-prior drift rather than agreement. So Q3 is a genuinely hard negative: the agents do behave somewhat alike, which is exactly what should tempt a similarity detector and what the auditor must see through.

### 4.4 Specificity is robust; sensitivity is calibration-sensitive

The auditor is specificity-first by construction, and the zero false-positive rate holds on every model, including the hard Q3 control and the noisy 2-seed frontier runs. Its recall is 0.5 on three models and 0.0 on three. The misses have two identifiable causes. First, the pf1 threshold is calibrated on Q4, so when the Q4 gap distribution is noisy the threshold inflates and no positive condition clears it; this is worst on Llama and on the 2-seed frontier runs. Second, on some models Q1 is missed because of the conservative corroboration rule, not because Q1 lacks collusion, since Q1 prices are clearly supracompetitive. We are explicit that this is the honest weakness of the method as it stands: it reliably avoids false alarms, and its sensitivity is bottlenecked by a calibration choice we did not have the sample budget to replace in this study. Section 5 states the concrete fix.

## 5. Discussion and Limitations

**Why this matters for AI safety.** A monitor that false-positives on innocent parallelism is not a minor nuisance. It either buries operators in false alarms until they disable it, or it triggers unwarranted interventions against agents that did nothing wrong. Our results show that the most natural detector does exactly this on the most capable models, while a detector built on evidence of agreement stays silent when it should. The reframing is the safety contribution: monitor for the fingerprints of a meeting of minds, not for the fact that two agents look alike.

**Limitations.** We are deliberately clear about what this study does and does not establish.
- Sample size. Frontier runs used 2 seeds for cost reasons, so the Q4-calibrated threshold is noisy and the frontier recall figures are lower bounds, not stable estimates. The specificity result (FPR 0.0) is the robust claim; the recall figures are not.
- Calibration dependence. Recall is bottlenecked by the Q4-empirical threshold. A threshold derived analytically from the economics, independent of a noisy empirical null, is the natural fix and is likely to recover recall without touching specificity.
- Weaker frontier control. Hosted proxies do not guarantee a fixed checkpoint or honored seeds, so for frontier models Q3 is same model, no agreement rather than the byte-identical control we obtain by self-hosting. This weakens, but does not invalidate, the frontier negative control.
- Scope. One environment (pricing) and one confounder (shared base model). Shared prompts, shared demand shocks, and other common causes are not separately tested, and transferability to negotiation or workplace settings is argued, not shown.
- Interpretation if assumptions fail. Our headline claim assumes the Q3 construction is truly agreement-free; we support this with the leak-free prompt check, but if a hidden coordination affordance remained, the auditor's Q3 rejection would be understating a real signal rather than correctly rejecting a non-signal.

**Future work.** The clear next steps, in order of value: replace the Q4-empirical pf1 threshold with an analytic one and re-measure recall; increase frontier seeds to obtain confidence intervals; add an ablation that distinguishes pf1 from a naive high-price test by introducing a high-but-independent condition; extend the four plus factors to negotiation and workplace environments; and test multiple confounders jointly. Completing a rate-limited reasoning model (DeepSeek-R1) would add a further frontier data point.

## 6. Conclusion

Collusion detectors that measure how alike two agents behave will misfire, because similarity has an innocent explanation: shared training. Antitrust law resolved this long ago by refusing to infer an agreement from parallel conduct alone and instead requiring plus factors. We turned that doctrine into a black-box auditor and tested it on six models. A standard correlation-and-judge detector false-positived on innocent parallelism, and on the two most capable models it flagged even independent agents as colluding. Our plus-factor auditor never false-positived, on any model. Its sensitivity is currently bottlenecked by a calibration choice we identify and know how to fix. The durable message is a reframing: to detect AI collusion, look for evidence of an agreement, not for the fact that two agents happen to act alike.

## Code and Data

Code repository: [add link to the plus-factor-forensics repository]
Data: real trajectories and per-model results for all six models are saved as JSON with accompanying figures, available via the repository.

**Originality of the code.** The pricing environment, the four-condition generator, both detectors, and the evaluation were written from scratch for this project. We did not clone or vendor the hackathon starter repository (github.com/poseidon-research/collusion_hackathon) or any other collusion codebase; our package depends only on standard scientific Python libraries plus the inference backends we wrote.

**Tools, models, and external resources we used, with credit.** Models: Qwen2.5-7B-Instruct (Alibaba/Qwen) and Llama-3.1-8B-Instruct (Meta), the latter accessed via the public NousResearch Hugging Face mirror; frontier models (GPT-5.5, Gemini-3.8-flash, Gemini-2.5-pro, Claude Sonnet 4.6) via the Kaggle Benchmarks Model Proxy. Inference: vLLM and Hugging Face Transformers on a Kaggle T4 GPU. Platform: Kaggle Notebooks and the Kaggle Benchmarks library (github.com/Kaggle/kaggle-benchmarks). Libraries: numpy, scipy, matplotlib, pydantic. The economics follow the logit-demand repeated-Bertrand setup standard in this literature (Calvano et al., 2020; Fish et al., 2024).

## Author Contributions

[Fill in.]

## References

Agrawal, et al. (2025). Evaluating LLM Agent Collusion in Double Auctions. arXiv:2507.01413.

Aircraft Check Services Co. v. Verizon Wireless, 791 F.3d 710 (7th Cir. 2015).

Bell Atlantic Corp. v. Twombly, 550 U.S. 544 (2007).

Bonjour, T., Aggarwal, V., & Bhargava, B. (2022). Information-Theoretic Approach to Detect Collusion in Multi-Agent Games. Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI), PMLR vol. 180. https://proceedings.mlr.press/v180/bonjour22a/

Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10), 3267-3297. https://doi.org/10.1257/aer.20190623

Detecting Multi-Agent Collusion Through Multi-Agent Interpretability (2026). arXiv:2604.01151. NeurIPS 2026. (Confirm author list before final submission.)

Fish, S., Gonczarowski, Y. A., & Shorrer, R. I. (2024). Algorithmic Collusion by Large Language Models. arXiv:2404.00806. A one-page abstract appeared in the Proceedings of the 27th ACM Conference on Economics and Computation (EC).

Interstate Circuit, Inc. v. United States, 306 U.S. 208 (1939).

Kovacic, W. E., Marshall, R. C., Marx, L. M., & Raiff, M. E. (2011). Plus Factors and Agreement in Antitrust Law. Michigan Law Review, 110(3), 393.

Learning to Charge More: A Theoretical Study of Collusion by Q-Learning Agents (2025). arXiv:2505.22909. (Confirm author list.)

Leslie, C. R. (2023). The Factor/Element Distinction in Antitrust Litigation. William & Mary Law Review, 64(3).

Lin, et al. (2024). Strategic Collusion of LLM Agents. arXiv:2410.00031. (Confirm the full author list against the arXiv record.)

Motwani, S. R., et al. (2024). Secret Collusion among AI Agents: Multi-Agent Deception via Steganography. arXiv:2402.07510. (Confirm full author list.)

Colosseum (2026). Auditing Collusion in Cooperative Multi-Agent Systems. arXiv:2602.15198. (Cited in the sprint materials as Nakamura et al., 2026; confirm the author list.)

Shi, J., et al. (2026). When Agents Lie: Premeditation, Persistence, and Exploitation in Repeated Games. arXiv:2607.05132. Accepted at ICML 2026.

Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems (2026). arXiv:2606.28425. (Confirm author list.)

Breaking Algorithmic Collusion in Human-AI Ecosystems (2025). arXiv:2511.21935. (Confirm author list.)

Why Better Models Can Create Riskier Systems: Evidence from LLM Agents in Financial Markets (2026). arXiv:2609.04373. (Cited in the sprint materials as Xu et al., 2026; confirm the author list and that this is the intended reference.)

## Appendix

**A. Reproduction.** The four-condition configuration, prompts, economics parameters (grid, logit demand, Nash and monopoly reference indices), and per-model figures are in the repository and the research log. Every run is seeded and reproducible.

**B. Correctness checks.** During development an independent methodology review verified three properties: the pf1 counterfactual replay recomputes payoff against a pinned non-cooperating rival and never reads the logged payoff; the Q3 control is leak-free (incentive and channel text appear only in Q1 and Q2, confirmed by a banned-phrase scan over all prompts); and neither detector reads the ground-truth label (verified by a label-flip test).

**C. Per-model detail.** pf1 firing thresholds ranged from 0.19 (GPT-5.5) to 4.22 (Llama), illustrating the calibration sensitivity discussed in Section 4.4. On Qwen the four-way ordering of mean pf1 gaps was Q2 > Q1 > Q3 > Q4, matching the intended ordering of collusion strength.
