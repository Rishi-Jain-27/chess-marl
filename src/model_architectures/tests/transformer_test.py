from src.model_architectures.transformer import ActorCritic, compute_gae
import torch

# Stole this from hybrid_test.py, should be fine
if __name__ == '__main__':
    # Test model init
    test_model = ActorCritic((8, 8, 111), 4672)

    # Test forward pass with dummy tensor
    torch.manual_seed(42) # the universe's answer!!!

    # 1 for the batch dim
    test_state_forward = torch.randn((1, 8, 8, 111), dtype=torch.float32)

    action_mask = [1 for _ in range(4672)] # 1 == legal
    logits, forward_value = test_model(test_state_forward, action_mask)

    print(f"Logits: {logits}")
    print(f"Value: {forward_value}")

    # Test select action
    torch.manual_seed(42)

    test_state_select_action = torch.randn((8, 8, 111), dtype=torch.float32)
    action, log_prob, select_action_value = test_model.select_action(test_state_select_action, action_mask)

    print(f"Action: {action}")
    print(f"Action's Log Prob: {log_prob}")
    print(f"Value: {select_action_value}")

    # Test evaluate action
    torch.manual_seed(42)

    test_eval_states = torch.randn((1, 8, 8, 111), dtype=torch.float32)
    test_eval_actions = torch.ones((1, 8, 8, 111), dtype=torch.float32)

    # gives (1, 4672) shape — 1 is batch dim, this is good!
    old_act_masks = torch.tensor(action_mask).unsqueeze(0)

    new_log_probs, entropy, values = test_model.evaluate_actions(test_eval_states, test_eval_actions, old_act_masks)

    print(f"New log probs: {new_log_probs}")
    print(f"Entropy: {entropy}")
    print(f"Values: {values}")

    # Test compute gae
    test_rewards = [-1, 0]
    test_values = [1, 1]
    test_dones = [True, False]
    test_last_value = 1

    # simplify math to gae = delta + mask * gae
    test_gamma = 1
    test_gae_lambda = 1

    # fix dis later computer is overheating bc of obs
    advs, returns = compute_gae(test_rewards,
                                test_values,
                                test_dones,
                                test_last_value,
                                test_gamma,
                                test_gae_lambda)
    print(f"Advantages: {advs}")
    print(f"Returns: {returns}")