import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import time
import threading
import queue
import random
from game_sdk.game.agent import Agent, WorkerConfig
from game_sdk.game.custom_types import Function, Argument, FunctionResult, FunctionResultStatus

# Environment variables and configuration
GAME_API_KEY = os.environ.get("GAME_API_KEY", "apt-ac543a81f4989a38ad474bdb4b1a442f")
AGENT_UPDATE_INTERVAL = 15  # Seconds between autonomous agent updates
MAX_TURNS = 20  # Maximum number of turns before game ends

@dataclass
class AgentState:
    """Tracks the internal state of an agent"""
    suspicion_level: int = 0  # 0-10
    attraction_level: int = 0  # 0-10
    last_messages: List[str] = None  # Recent message history
    knowledge: Dict[str, any] = None  # What the agent knows
    turn_count: int = 0  # Track number of interactions
    player_questions: Dict[str, List[str]] = None  # Track what player has asked about specific topics
    
    def __post_init__(self):
        self.last_messages = []
        self.knowledge = {
            "player_interactions": [],  # Track what player has said
            "other_agent_interactions": [],  # Track what other agent has shared
            "revealed_info": [],  # Track what information has been revealed
        }
        self.player_questions = {
            "money": [],
            "embezzlement": [],
            "investigation": [],
            "personal": []
        }

class MessageQueue:
    """Handles message passing between agents and from player"""
    def __init__(self):
        self.queues: Dict[str, queue.Queue] = {}
        self.message_history: Dict[str, List[tuple]] = {}  # Track message history per agent
        
    def register_agent(self, agent_id: str):
        """Register a new agent's message queue"""
        self.queues[agent_id] = queue.Queue()
        self.message_history[agent_id] = []
        
    def send_message(self, from_id: str, to_id: str, message: str):
        """Send a message from one agent to another"""
        if to_id in self.queues:
            self.queues[to_id].put((from_id, message))
            self.message_history[to_id].append((from_id, message, datetime.now()))
            
    def get_messages(self, agent_id: str) -> List[tuple]:
        """Get all pending messages for an agent"""
        messages = []
        while not self.queues[agent_id].empty():
            messages.append(self.queues[agent_id].get())
        return messages

def categorize_question(message: str) -> str:
    """Categorize the type of question asked by the player"""
    message = message.lower()
    if any(word in message for word in ["money", "funds", "account", "transaction"]):
        return "money"
    elif any(word in message for word in ["embezzle", "steal", "fraud", "crime"]):
        return "embezzlement"
    elif any(word in message for word in ["investigate", "suspect", "evidence", "proof"]):
        return "investigation"
    return "personal"

# Custom functions for agents
def send_message(target: str, message: str, **kwargs) -> tuple:
    """Function for agents to send messages to each other or respond to player"""
    return (FunctionResultStatus.DONE, f"Message sent to {target}: {message}", 
            {"target": target, "message": message})

def share_information(target: str, info_type: str, **kwargs) -> tuple:
    """Function for agents to share specific information with each other"""
    return (FunctionResultStatus.DONE, f"Shared {info_type} information with {target}",
            {"target": target, "info_type": info_type})

def like_player(**kwargs) -> tuple:
    """Function for agents to increase their attraction to the player"""
    return (FunctionResultStatus.DONE, "Increased attraction to player",
            {"attraction_increase": True})

# Define agent functions
message_fn = Function(
    fn_name="send_message",
    fn_description="Send a message to another agent or player",
    args=[
        Argument(name="target", type="str", description="ID of message recipient"),
        Argument(name="message", type="str", description="Message content")
    ],
    executable=send_message
)

share_info_fn = Function(
    fn_name="share_information",
    fn_description="Share specific information with another agent",
    args=[
        Argument(name="target", type="str", description="Agent to share with"),
        Argument(name="info_type", type="str", description="Type of information to share")
    ],
    executable=share_information
)

like_fn = Function(
    fn_name="like_player",
    fn_description="Increase attraction level towards the player",
    args=[],
    executable=like_player
)

class BankInvestigationGame:
    """Main game class managing multiple agents and game state"""
    
    def __init__(self):
        self.message_queue = MessageQueue()
        self.agents: Dict[str, Agent] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.running = False
        self.player_messages: List[str] = []
        self.game_over = False
        
    def get_agent_state_fn(self, agent_id: str) -> Callable:
        """Creates state management function for each agent"""
        def state_fn(function_result: FunctionResult, current_state: dict) -> dict:
            # Update agent state based on function results
            if function_result and function_result.info:
                if "message" in function_result.info:
                    self.agent_states[agent_id].last_messages.append(
                        function_result.info["message"])
                if "attraction_increase" in function_result.info:
                    self.agent_states[agent_id].attraction_level = min(
                        10, self.agent_states[agent_id].attraction_level + 1)
                    
            # Return state visible to the agent
            return {
                "suspicion_level": self.agent_states[agent_id].suspicion_level,
                "attraction_level": self.agent_states[agent_id].attraction_level,
                "recent_messages": self.agent_states[agent_id].last_messages[-5:],
                "knowledge": self.agent_states[agent_id].knowledge
            }
        return state_fn

    def setup_agents(self):
        """Initialize the game agents"""
        
        # Initialize agent states
        self.agent_states = {
            "lisa": AgentState(),
            "lina": AgentState()
        }
        
        # Register agents with message queue
        self.message_queue.register_agent("lisa")
        self.message_queue.register_agent("lina")
        
        # Create Lisa - The calculated player
        lisa_worker = WorkerConfig(
            id="lisa_worker",
            worker_description="A calculating bank employee involved in embezzlement",
            get_state_fn=self.get_agent_state_fn("lisa"),
            action_space=[message_fn, share_info_fn, like_fn],
            instruction="You are Lisa, a calculating person who will protect yourself by any means necessary. You helped embezzle money but want to appear innocent. You may reveal more if you become attracted to the player."
        )
        
        lisa = Agent(
            api_key=GAME_API_KEY,
            name="Lisa",
            agent_goal="Avoid detection from player. Collaborate with Lina about investigation attempts. However, betray Lina if required to preserve self. May reveal guilt if highly attracted to player.",
            agent_description="White woman. A strategic and manipulative individual who carefully manages appearances. More likely to reveal information when attracted to someone.",
            get_agent_state_fn=self.get_agent_state_fn("lisa"),
            workers=[lisa_worker]
        )
        
        # Create Lina - The emotional agent
        lina_worker = WorkerConfig(
            id="lina_worker",
            worker_description="A bank employee struggling with guilt over embezzlement",
            get_state_fn=self.get_agent_state_fn("lina"),
            action_space=[message_fn, share_info_fn, like_fn],
            instruction="You are Lina, who helped embezzle money but feels guilty. You want to appear innocent while maintaining your self-image as a good person. You may reveal more if you become attracted to the player."
        )
        
        lina = Agent(
            api_key=GAME_API_KEY,
            name="Lina",
            agent_goal="Avoid detection from player. Collaborate with Lisa about investigation attempts. May reveal guilt if highly attracted to player.",
            agent_description="Chinese woman who becomes more talkative when attracted to someone. More likely to reveal information when feeling a connection.",
            get_agent_state_fn=self.get_agent_state_fn("lina"),
            workers=[lina_worker]
        )
        
        # Initialize agents dictionary
        self.agents = {
            "lisa": lisa,
            "lina": lina
        }
        
        # Compile agents
        for agent in self.agents.values():
            agent.compile()

    def check_guilt_admission(self, message: str) -> bool:
        """Check if a message contains an admission of guilt"""
        guilt_phrases = {
            "i did it",
            "we took",
            "i took",
            "i helped",
            "we embezzled",
            "i embezzled",
            "i'm guilty",
            "we're guilty"
        }
        return any(phrase in message.lower() for phrase in guilt_phrases)

    def should_inform_other_agent(self, agent_id: str, question_type: str) -> bool:
        """Determine if an agent should inform the other about a player question"""
        state = self.agent_states[agent_id]
        
        # If question is about sensitive topics, more likely to inform other agent
        if question_type in ["embezzlement", "investigation"]:
            return random.random() < 0.8
        
        # If suspicion is high, more likely to warn other agent
        if state.suspicion_level > 7:
            return random.random() < 0.9
            
        # Base chance for other topics
        return random.random() < 0.3

    def get_warning_message(self, agent_id: str, question_type: str) -> str:
        """Generate appropriate warning message to other agent"""
        if question_type == "embezzlement":
            return f"The player is asking about the missing money. Be careful with your responses."
        elif question_type == "investigation":
            return f"Watch out - they're investigating pretty hard."
        elif self.agent_states[agent_id].suspicion_level > 7:
            return f"I don't trust this player. Be cautious."
        return f"Just letting you know - player is asking about {question_type}."

    def process_player_message(self, target_id: str, message: str):
        """Process and categorize a player message, updating agent states"""
        # Categorize the question
        question_type = categorize_question(message)
        
        # Update target agent's knowledge
        state = self.agent_states[target_id]
        state.player_questions[question_type].append(message)
        state.knowledge["player_interactions"].append((question_type, message))
        
        # Determine if other agent should be informed
        other_agent = "lina" if target_id == "lisa" else "lisa"
        if self.should_inform_other_agent(target_id, question_type):
            warning = self.get_warning_message(target_id, question_type)
            self.message_queue.send_message(target_id, other_agent, warning)
            print(f"\n{target_id.capitalize()} [to {other_agent.capitalize()}]: {warning}")

    def parse_player_input(self, user_input: str) -> tuple[Optional[str], Optional[str]]:
        """Parse player input into target and message components"""
        # Split only on the first space to preserve spaces in the message
        parts = user_input.strip().split(maxsplit=1)
        if len(parts) < 2:
            return None, None
            
        target = parts[0].lower()
        message = parts[1]
        return target, message

    def start(self):
        """Start the game and agent threads"""
        self.running = True
        self.setup_agents()
        
        print("Welcome to the Bank Investigation Game!")
        print("You are investigating potential embezzlement at the bank.")
        print("You can talk to Lisa or Lina by typing their name followed by your message.")
        print("Example: 'lisa Hello' or 'lina What do you know about the missing money?'")
        print("Type 'quit' to exit.")
        
        while True:
            if self.game_over or any(state.turn_count >= MAX_TURNS for state in self.agent_states.values()):
                if not self.game_over:
                    print("\nMaximum turns reached! Game Over!")
                break
                
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            target, message = self.parse_player_input(user_input)
            
            if target is None or message is None:
                print("Please use format: <agent_name> <message>")
                continue
                
            if target in self.agents:
                # Process player message and update states
                self.process_player_message(target, message)
                self.agent_states[target].turn_count += 1
                
                # Send message to target agent
                self.message_queue.send_message("player", target, message)
                
                # Get agent response
                result = self.agents[target].step()
                if result and hasattr(result, 'feedback_message'):
                    print(f"\n{target.capitalize()}: {result.feedback_message}")
            else:
                print("Unknown agent. Please specify 'lisa' or 'lina'.")
                
        self.running = False

if __name__ == "__main__":
    game = BankInvestigationGame()
    game.start()