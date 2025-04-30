import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification

import torch.nn.functional as F
import json
from ferret.evaluators import BaseEvaluator
from ferret.explainers.explanation import Explanation
from ferret.evaluators.evaluation import Evaluation
from src.metrics_util import *


def mass_acc(explanation: Explanation, mask) -> float:
    if type(mask) != list:
        mask = [int(m) for m in json.loads(mask)]
    normalized = explanation.scores / np.sum(explanation.scores)
    return np.sum(normalized[mask])


def gini_index(explanation: Explanation, eps=1e-12) -> float:
    """
    For an attribution method A, we quantify the sparseness of
    the attribution vector using the Gini Index applied
    to the vector of absolute values
    > https://proceedings.mlr.press/v119/chalasani20a/chalasani20a.pdf

    Lies in [0,1]; higher is better (more sparse).

    There are different implementations that does not get the absolute values
        but shift the min. to zero to ensure non-negativity
        we followed the paper above and got the absolute values
    """
    # get absolute value and add eps so no zeros allowed
    v = np.sort(np.abs(explanation.scores) + eps)
    k = np.arange(1, v.shape[0] + 1)  # sorted indices
    d = v.shape[0]  # size
    return 1 - 2 * np.sum((v / np.linalg.norm(v, ord=1)) * (1 - k / d + 0.5 / d))


def threshold_sparsity(explanation: Explanation, t=0.01) -> float:
    """
    Get the share of scores in an attribution vector with
        absolute value higher than the given threshold
    """
    n = explanation.scores.shape[0]
    return np.sum(np.abs(explanation.scores) >= t) / n


from ferret.evaluators import BaseEvaluator
import torch.nn.functional as F


class EvaluationMetricOutput:
    """Output to store any metric result."""

    metric: BaseEvaluator
    value: float


class Sensitivity_Evaluation(BaseEvaluator):
    NAME = "auc_sensitivity"
    SHORT_NAME = "sens"

    LOWER_IS_BETTER = True
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    BEST_VALUE = 0.0
    METRIC_FAMILY = "faithfulness"

    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "faithfulness"

    def compute_evaluation(self, explanation, target=1, **evaluation_args):
        """Evaluate an explanation on the Sensitivity metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args.
                We currently support multiple approaches to define the hard rationale from
                soft score rationales, based on:
                - th : token greater than a threshold
                - perc : more than x% of the tokens
                - k: top k values

        Returns:
            Evaluation : the Sensitivity score of the explanation
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Parsing additional evaluation arguments
        remove_first_last, _, _, top_k, name_model = parse_evaluator_args_model(
            evaluation_args
        )

        removal_args = {"remove_tokens": True, "based_on": "k"}
        if isinstance(explanation, list):
            return None

        text = explanation.text
        tokenizer = self.helper.tokenizer
        score_explanation = explanation.scores

        removal_args["remove_tokens"] = True

        # Tokenize the input text
        item = self.helper._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        # Remove first and last tokens if specified
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        # Define thresholds and epsilon values for evaluation
        n = input_len - 1
        n = min(n, top_k)
        thresholds = [i for i in range(1, n)]
        epsilons = []

        # Load classification model and base model
        name = name_model
        # classification_model = AutoModelForSequenceClassification.from_pretrained(name)
        classification_model = self.helper.model.to(device)  # for fine-tuned weights
        base_model = AutoModel.from_pretrained(name).to(device)

        # Tokenize text for the model
        inputs = tokenizer(text, return_tensors="pt").to(device)
        if remove_first_last == True:
            inputs = {
                "input_ids": inputs["input_ids"][:, 1:-1],
                "attention_mask": inputs["attention_mask"][:, 1:-1],
            }

        # Get token embeddings
        with torch.no_grad():
            outputs = base_model(**inputs)
            token_embeddings = outputs.last_hidden_state

        for v in thresholds:
            # Get rationale perturbation vector
            perturbation_vector = get_discrete_explanation_topK(
                score_explanation, v, only_pos=False
            )
            # print(f"Threshold: {v}, Perturbation Vector: {perturbation_vector}")

            # Define attack parameters
            alpha = 0.1
            num_steps = 5

            # Get original model prediction
            with torch.no_grad():
                original_logits = classification_model(**inputs).logits
                y = torch.argmax(original_logits, dim=-1)

            # Define PGD attack function
            def pgd_attack(
                embeddings, epsilon, alpha, num_steps, perturbation_vector, y
            ):
                perturbed_embeddings = embeddings.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([perturbed_embeddings], lr=alpha)

                for step in range(num_steps):
                    optimizer.zero_grad()
                    classificator_inputs = {
                        "attention_mask": inputs["attention_mask"],
                        "inputs_embeds": perturbed_embeddings,
                    }
                    logits = classification_model(**classificator_inputs).logits
                    loss = F.cross_entropy(logits, y)
                    loss.backward()

                    with torch.no_grad():
                        grad = perturbed_embeddings.grad.clone().to(device)
                        mask = (
                            torch.tensor(perturbation_vector, dtype=torch.float32)
                            .unsqueeze(0)
                            .unsqueeze(-1)
                            .to(device)
                        )
                        mask = mask.expand_as(grad)
                        grad = grad * mask
                        perturbed_embeddings += alpha * grad.sign()
                        perturbation = torch.clamp(
                            perturbed_embeddings - embeddings, -epsilon, epsilon
                        )
                        perturbed_embeddings = (
                            (embeddings + perturbation).detach().requires_grad_(True)
                        )

                    classification_model.zero_grad()
                    if perturbed_embeddings.grad is not None:
                        perturbed_embeddings.grad.zero_()

                return perturbed_embeddings

            # Binary search for optimal epsilon value
            def binary_search_epsilon(
                token_embeddings,
                perturbation_vector,
                alpha,
                num_steps,
                y,
                tol=1e-3,
                max_iter=15,
            ):
                low = 0.0
                high = 1.0
                best_epsilon = low

                for i in range(max_iter):
                    mid = (low + high) / 2.0
                    perturbed_embeddings = pgd_attack(
                        token_embeddings, mid, alpha, num_steps, perturbation_vector, y
                    )
                    classificator_inputs = {
                        "attention_mask": inputs["attention_mask"],
                        "inputs_embeds": perturbed_embeddings,
                    }

                    with torch.no_grad():
                        classification_outputs = classification_model(
                            **classificator_inputs
                        )
                        logits = classification_outputs.logits
                        predictions = torch.argmax(logits, dim=-1)

                    if predictions != y:
                        best_epsilon = mid
                        high = mid
                    else:
                        low = mid

                    if high - low < tol:
                        break

                return best_epsilon

            # Compute epsilon for current threshold
            epsilon = binary_search_epsilon(
                token_embeddings, perturbation_vector, alpha, num_steps, y
            )
            epsilons.append(epsilon)

        # Compute AUC for sensitivity
        sens_auc = np.trapz(epsilons, thresholds)
        # print(f"Epsilons: {epsilons}, Sensitivity AUC: {sens_auc}")

        evaluation_output = Evaluation(self.SHORT_NAME, sens_auc)
        return evaluation_output
