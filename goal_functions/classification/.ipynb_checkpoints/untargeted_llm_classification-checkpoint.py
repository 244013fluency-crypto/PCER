"""

Determine successful in untargeted Classification
----------------------------------------------------
"""

from .classification_goal_function import ClassificationGoalFunction


class UntargetedLLMClassification(ClassificationGoalFunction):
    """An untargeted attack on classification models which attempts to minimize
    the score of the correct label until it is no longer the predicted label.

    Args:
        target_max_score (float): If set, goal is to reduce model output to
            below this score. Otherwise, goal is to change the overall predicted
            class.
    """

    def __init__(self, *args, inference, target_max_score=None, **kwargs):
        self.target_max_score = target_max_score
        self.inference = inference
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        if self.target_max_score:
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(
                self.ground_truth_output, float
        ):
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output, _):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            return 1 - model_output[self.ground_truth_output]

    def _call_model(self, attacked_text_list):
        acc_list = []
        for text in attacked_text_list:
            # prompt = prompt.text
            print("Current attacked text is: {}".format(text))
            acc = self.inference.predict(text)  # 一个列表置信度得分的列表[]
            print("Current acc: {:.2f}".format(acc * 100))
            acc_list.append(acc)
        return self._process_model_outputs(attacked_text_list, acc_list)
#     def create_goal_function(args, inference_model):
#     goal_function = PromptGoalFunction(inference=inference_model,
#                                        query_budget=args.query_budget,
#                                        logger=args.logger,
#                                        model_wrapper=None,
#                                        verbose=args.verbose)
#     return goal_function

# from inference import Inference
# inference_model = Inference(args)
