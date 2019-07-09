#### Frequentist v.s. Bayesian

Usually, we face problems that given samples to find out the the hypothesis conditions (parameters). In statistics, that is to maximize:
$$
P(Hypotheses|Samples)
$$
In general, there're 2 major parties for this type of question: Frequentist and Bayesian. 

##### Frequentist

Frequentist inference based on likelihood, and does not require a prior hypothesis on probability, which is actually the MLE routine, Maximum Likelihood Estimation.
$$
L(\theta|x_i) = \prod_{i=1} ^n P(x_i|\theta)
$$
To maximize the likelihood is to maximize its 'log' transformation:
$$
log(L(\theta|x_i))=log(\prod_i P(x_i|\theta))=\sum_i log(P(x_i|\theta))
$$
take partials with respect to θ and set it equal to 0:


$$
\frac {\part log(L(\theta|x_i))} {\part \theta} = \frac {\part log(\prod_i P(x_i|\theta))}{\part \theta}=\frac {\part \sum_i logP(x_i|\theta)}{\part \theta} = 0
$$
simplify the function and get the MLE estimator for θ. 

Therefore, for MLE estimator:
$$
\begin{align}
\theta_{MLE} &=argmax_\theta P(X|\theta)\\
&=argmax_\theta logP(X|\theta)\\
&= argmax_\theta log\prod_iP(x_i | \theta)\\
&=argmax_\theta \sum_i logP(x_i|\theta)
\end{align}
$$


##### Bayesian

Bayesian inference is based on Bayes' Theorem, where it considers about probabilities for both samples and hypotheses, thus requires the probability for prior hypothesis. It follows a MAP (Maximum A Posteriori Estimation) routine.
$$
\begin{align}
\theta_{MAP} &=argmax_\theta P(X|\theta)P(\theta) \\
&=argmax_\theta logP(X|\theta)P(\theta)\\
&=argmax_\theta log \prod_i P(x_i|\theta)P(\theta)\\
&=argmax_\theta \sum_i logP(x_i|\theta)P(\theta)
\end{align}
$$
