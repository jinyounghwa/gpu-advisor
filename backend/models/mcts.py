"""
MCTS (Monte Carlo Tree Search) 구현
AlphaGo/MuZero 핵심 탐색 알고리즘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MCTSConfig:
    """MCTS 하이퍼파라미터"""
    num_simulations: int = 50
    exploration: float = 1.5
    dirichlet_alpha: float = 0.25
    rollout_steps: int = 5
    temperature: float = 1.0


@dataclass
class Node:
    """MCTS 트리 노드"""
    state: Optional[torch.Tensor] = None  # Latent State (1, 256)
    visits: int = 0
    total_reward: float = 0.0
    parent: Optional['Node'] = None
    action: Optional[int] = None
    prior: float = 1.0
    children: Dict[int, 'Node'] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


class MCTSTree:
    """MuZero 스타일 MCTS 트리"""

    def __init__(self, config: MCTSConfig, action_dim: int = 5):
        self.config = config
        self.action_dim = action_dim

    def search(self, root_state: torch.Tensor, dynamics_model: nn.Module, prediction_model: nn.Module) -> Tuple[int, torch.Tensor]:
        """
        MCTS 탐색 수행
        """
        root = Node(state=root_state)
        
        # Root expansion
        self._expand_node(root, prediction_model)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # 1. Selection
            while node.children and all(child.state is not None for child in node.children.values()):
                node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion & Evaluation
            # Selection이 멈춘 지점(node)의 자식 중 하나를 선택해 확장
            if node.children:
                # 아직 state가 없는 자식들 중 하나 선택 (여기선 단순화해서 PUCT로 선택 가능하도록 임시 state 부여하거나 순차 선택)
                # 우선순위가 높은(PUCT) 자식 중 state가 없는 녀석 선택
                best_action = -1
                best_score = -float('inf')
                for a, child in node.children.items():
                    if child.state is None:
                        # PUCT score (visits=0 이므로 prior가 중요)
                        score = child.prior * (math.sqrt(max(1, node.visits)) / (1 + child.visits))
                        if score > best_score:
                            best_score = score
                            best_action = a
                
                target_node = node.children[best_action]
                search_path.append(target_node)
                
                # Dynamics + Prediction
                action_onehot = torch.zeros(1, self.action_dim, device=root_state.device)
                action_onehot[0, best_action] = 1.0
                
                with torch.no_grad():
                    next_state, reward_mean, _ = dynamics_model(node.state, action_onehot)
                    target_node.state = next_state
                    self._expand_node(target_node, prediction_model)
                    _, value = prediction_model(next_state)
                    leaf_value = value.item()
            else:
                # 자식이 없는 경우 (이미 다 확장됨 - 사실 위 로직에선 발생 안함)
                with torch.no_grad():
                    _, value = prediction_model(node.state)
                    leaf_value = value.item()

            # 3. Backpropagation
            self._backpropagate(search_path, leaf_value)

        # 최적 액션 선택 (방문 횟수 기준)
        visit_counts = torch.tensor([root.children[a].visits for a in range(self.action_dim)], dtype=torch.float32)
        policy_target = visit_counts / visit_counts.sum()
        best_action = torch.argmax(visit_counts).item()

        return best_action, policy_target

    def _expand_node(self, node: Node, prediction_model: nn.Module):
        """노드 확장: 정책 확률(Prior) 할당"""
        with torch.no_grad():
            policy_logits, _ = prediction_model(node.state)
            probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        
        for a in range(self.action_dim):
            node.children[a] = Node(parent=node, action=a, prior=probs[a].item())

    def _select_child(self, node: Node) -> Node:
        """PUCT 알고리즘으로 자식 선택"""
        total_sqrt = math.sqrt(node.visits)
        best_score = -float('inf')
        best_child = None

        for action, child in node.children.items():
            u_score = self.config.exploration * child.prior * (total_sqrt / (1 + child.visits))
            score = child.value + u_score

            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def _backpropagate(self, search_path: List[Node], value: float):
        for node in reversed(search_path):
            node.visits += 1
            node.total_reward += value


if __name__ == "__main__":
    # Test MCTS with dummy models
    from representation_network import RepresentationNetwork
    from dynamics_network import DynamicsNetwork
    from prediction_network import PredictionNetwork

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    latent_dim = 256
    action_dim = 5
    
    # Initialize models
    h = RepresentationNetwork(state_dim=22, latent_dim=latent_dim).to(device)
    g = DynamicsNetwork(latent_dim=latent_dim, action_dim=action_dim).to(device)
    f = PredictionNetwork(latent_dim=latent_dim, action_dim=action_dim).to(device)
    
    # Config
    config = MCTSConfig(num_simulations=50)
    tree = MCTSTree(config, action_dim=action_dim)
    
    # Initial state
    root_observation = torch.randn(1, 22).to(device)
    root_latent = h(root_observation)
    
    print("Starting MCTS search...")
    best_action, policy_target = tree.search(root_latent, g, f)
    
    print(f"Best action: {best_action}")
    print(f"Policy target: {policy_target}")
    print("MCTS Search completed successfully!")
