# Poster Session

Main result:
Rectified Flow and Transformers are _complementary_: distributional fidelity vs. single-gene accuracy.

Motivation:
- Drug is simply a molecule, perturbing a cell and having some desired effect on the cell
- If we knew what exactly a drug would change inside a cell, we could simulate the drug discovery process in silico
--> 
- Save time and resources and enable large speedup, by predicting suitable drug candidates with the desired effect 
- Test potential candidates in silico before having to test it in a time consuming and expensive process in the lab
- Accelerate Discoveries
- Overarching goal of the field: AI virtual cell (AIVC), a multi-scale, multi-modal large-neural-network-based model that can represent and simulate the behavior of molecules, cells, and tissues across diverse states (How to build a virtual cell)


Challenge:
- Molecule interacts in highly non-linear and complex ways with a cell. 
- Gene expression vectors vary significantly, even for cells within the same cell line and perturbation. 
- Distribution-to-Distribution problem rather than an input-output mapping

Idea 
- Many different architectures proposed. Most promising ones rely on flow-based models and transformers. 
- Compare Rectified Flow, as the simplest instantiation of flow-based approaches, with Transformers on 10 perturbations
- Perturbations have to be selected to represent different effect-sizes -> Get a more well rounded picture of model capabilities
	- 10 randomly selected perturbations might have very similar properties. Better: 10 perturbations with varying properties. 
- Establish understanding of each approaches strengths and weaknesses.
	- Differences between model predictions w.r.t. perturbation effect size -> One model better for small difference perturbations while the other is stronger for large differences compared to the control distribution?
	- Difference between metrics -> One model stronger w.r.t. E-distance while other model stronger w.r.t. DE-F1?
	- What is the result compared to mean? Where are the models outperforming the mean, where underperforming it? 
	- Maybe also log-fold change difference useful here? -> Know how closely the models got the true logfoldchange of the perturbation compared to the control cells right? 
- Guide future research, depending on the task, which architecture might be useful. 

*(Shorten these parts down to either one sentence or 2-3 bullet points)*

Introduction:
- *See above. Compile the introduction from this.* 

Methods:
- Dataset:
	- Tahoe-100M, SW480 colorectal cancer cell line, reduced to 10 perturbations. The dataset now has a total of 93677 perturbed cells and 15050 unperturbed control cells. 
- Rectified Flow:
	- Learns a transport flow from control cell distribution to perturbed distribution through a simple MLP approximation model
	- 
	- Formulas:
		- Training through interpolation estimation 
			- zt = (1− t) z0 + tz1
		- Standard MSE loss
			- L= Et∼ Beta(5,5)[||vθ(zt,t) − (z1− z0)||2]
		- Prediction step along the trajectory
			- zt+dt = zt + vθ(zt,t)dt

- Transformer:
	- Attention across cells within the same batch
	- Provide distributional understanding to Transformer
	- Transformer is further trained with E-distance 
	- *(No visualization here, because everybody seeing the poster probably understands on some level how a transformer works)*
	- *Attention mechanism formula can still be shown, maybe?*

- Baselines:
	- Mean baseline
		- Prediction is the mean of every perturbation from the training data (same access to data as models)
	- Mean shift baseline
		- Prediction is the average mean shift between the control cells and the perturbed cells.
		- x_shift = (y^train_perturbed - x^train_control)
		- x_mean-shift = x^test_control + x_shift
        *Maybe update the formulas here, as they are not very clean*

Metrics:
- DE-F1: biological validity (are the right genes called differentially expressed?)
- E-distance: distributional fidelity (does the whole distribution of predicted genes match the true perturbed distribution?)
- MSE: Gene-level accuracy of the predicted mean shift

Results:
\begin{table}[htbp]
    \centering
    \small
    \begin{tabular}{|c||c|c|c|c|c|}
        \hline
        Method & DE-F1 ($\uparrow$) & DE-F1 Precision ($\uparrow$) & DE-F1 Recall ($\uparrow$) & \ac{E-distance} ($\downarrow$) & \ac{MSE} ($\downarrow$) \\
        \hline
		Transformer & \textbf{0.503 $\pm$ 0.036} & \textbf{0.756 $\pm$ 0.054} & 0.410 $\pm$ 0.049 & 0.081 $\pm$ 0.006 & 0.335 $\pm$ 0.005 \\
		\hline
		Rectified Flow & 0.445 $\pm$ 0.026 & 0.604 $\pm$ 0.048 & 0.380 $\pm$ 0.030 & 0.068 $\pm$ 0.003 & 0.331 $\pm$ 0.002 \\
		\hline
		Mean & 0.094 $\pm$ 0.007 & 0.051 $\pm$ 0.004 & 0.914 $\pm$ 0.025 & 3.783 $\pm$ 0.004 & \textbf{0.166 $\pm$ 0.000} \\
		\hline
		Mean Shift & 0.122 $\pm$ 0.008 & 0.066 $\pm$ 0.005 & \textbf{0.967 $\pm$ 0.016} & \textbf{0.031 $\pm$ 0.001} & 0.335 $\pm$ 0.002 \\
        \hline
    \end{tabular}
    \caption{Performance comparison of models and baselines. Results report mean $\pm$ standard deviation over 20 independent runs. Best values per metric and perturbation are shown in \textbf{bold}.}
    \label{tab:pred_results}
\end{table}

Discussion
- Strong performance of models for DE-F1 score compared to baselines shows that models learn biologically meaningful gene expressions. 
- Mean shift baseline better at E-distance and Mean baseline better at MSE than models. This shows that simple baselines are able to fool metrics, but dont produce anything biologically meaningful, therefore multiple metrics are necessary. 
- Complementary pattern between Transformer and Rectified Flow
	- Transformer better for single gene predictions, especially relevant for DE-F1, while Rectified Flow superior for distributional fidelity
	- Models relevant for different tasks
- Models also complement each other w.r.t DE-F1 performance per perturbation. Perturbations where the Transformer has low score, Rectified Flow has significantly higher score, suggesting that some perturbations show stronger general distributional change, while others show stronger expression changes for very few genes, while the rest stays relatively unchanged. 
- DE-F1 score does not significantly increase with perturbation effect magnitude, while E-distance does increase strongly with effect magnitude, especially for Rectified Flow. We hypothesize this is due to stronger effect magnitude, meaning more single genes that need to be predicted as differentially expressed. Rectified Flow struggles, because it is better at predicting whole distributions instead of single genes (Transformer is better here). 

Future Work
- Test model performance on unseen perturbations and unseen cell lines. -> Test generalization capacity
- Combine Transformer with Rectified flow model to evaluate if complementary strengths can be leveraged this way

References
- *I will provide the references later*