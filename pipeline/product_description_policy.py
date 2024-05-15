from typing import Dict, Text, Any, Optional

from rasa.core.policies.policy import Policy
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.engine.storage.resource import Resource
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.core.constants import (
    POLICY_PRIORITY,
    POLICY_MAX_HISTORY,
)

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT], is_trainable=False
)
class ProductDescriptionPolicy(Policy):
    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        super().__init__(config, name, model_storage, resource)
        self.config = config
        self.name = name
        self.model_storage = model_storage
        self.resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            # Determines the importance of policies, higher values take precedence
            POLICY_PRIORITY: 0.0,
            POLICY_MAX_HISTORY: 10,
        }
    
    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        # check if we are in an active loop
        if tracker.active_loop:
            print("ProductDescriptionPolicy: Active loop detected")
        # print the latest intent detected
        print("ProductDescriptionPolicy: Latest intent detected: ", tracker.latest_message.get("intent", {}).get("name"))
        entites = tracker.latest_message.get("entities", [])
        if any(entity["entity"] == "product" for entity in entites):
            print("ProductDescriptionPolicy: Product entity detected")
            print("ProductDescriptionPolicy: Product entity value: ", next(entity["value"] for entity in entites if entity["entity"] == "product"))
        return PolicyPrediction([0.0 for _ in range(domain.num_actions)], "ProductDescriptionPolicy")
