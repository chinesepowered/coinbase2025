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
AGENT_UPDATE_INTERVAL = 6  # Seconds between autonomous agent updates
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
def reply_to_player(message: str, **kwargs) -> tuple:
    """Function for agents to respond to player queries"""
    return (FunctionResultStatus.DONE, f"Response to player: {message}", 
            {"message": message})

def share_information(target: str, info_type: str, suspicion_increase: int = 1, **kwargs) -> tuple:
    """Function for agents to share specific information with each other"""
    # Ensure suspicion_increase is an integer
    suspicion_increase = int(suspicion_increase)  # Convert to int if it's a string
    return (FunctionResultStatus.DONE, f"Shared {info_type} information with {target}",
            {"target": target, "info_type": info_type, "suspicion_increase": suspicion_increase})

def like_player(attraction_change: int = 1, **kwargs) -> tuple:
    """Function for agents to modify their attraction to the player"""
    # Ensure attraction_change is an integer
    attraction_change = int(attraction_change)  # Convert to int if it's a string
    return (FunctionResultStatus.DONE, 
            f"{'Increased' if attraction_change > 0 else 'Decreased'} attraction to player by {abs(attraction_change)}",
            {"attraction_change": attraction_change})

def do_nothing(**kwargs) -> tuple:
    """Function for agents when they choose to take no action"""
    return (FunctionResultStatus.DONE, "Agent chose to take no action", {})

# Define agent functions
reply_player_fn = Function(
    fn_name="reply_to_player",
    fn_description="Respond to a player's question or message. Not to be used for agents to communicate with each other, only in response to user questions direct to this agent.",
    args=[
        Argument(name="message", type="str", description="Response message to player")
    ],
    executable=reply_to_player
)

share_info_fn = Function(
    fn_name="share_information",
    fn_description="Share information with another agent about player's investigation, such as when player asks the agent a question that raises alarm and the agent wants to share that with the other agents to raise their suspcion as well.",
    args=[
        Argument(name="target", type="str", description="Agent to share with"),
        Argument(name="info_type", type="str", description="Type of information to share"),
        Argument(name="suspicion_increase", type="int", description="Amount to increase suspicion level (1-5)")
    ],
    executable=share_information
)

like_fn = Function(
    fn_name="like_player",
    fn_description="Modify attraction level towards the player",
    args=[
        Argument(name="attraction_change", type="int", description="Amount to change attraction (-8 to +8)")
    ],
    executable=like_player
)

do_nothing_fn = Function(
    fn_name="do_nothing",
    fn_description="Take no action this turn",
    args=[],
    executable=do_nothing
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
        self.last_step_time = 0  # Track when the last autonomous step occurred
        self.last_autonomous_time = 0  # Separate timer for autonomous updates
        self.autonomous_thread = None  # Thread for autonomous updates
        
    def get_agent_state_fn(self, agent_id: str) -> Callable:
        """Creates state management function for each agent"""
        def state_fn(function_result: FunctionResult, current_state: dict) -> dict:
            # Update agent state based on function results
            if function_result and function_result.info:
                if "message" in function_result.info:
                    self.agent_states[agent_id].last_messages.append(
                        function_result.info["message"])
                if "attraction_change" in function_result.info:
                    self.agent_states[agent_id].attraction_level = max(0, min(10,
                        self.agent_states[agent_id].attraction_level + function_result.info["attraction_change"]))
                if "suspicion_increase" in function_result.info:
                    other_agent = "lina" if agent_id == "lisa" else "lisa"
                    self.agent_states[other_agent].suspicion_level = max(0, min(10,
                        self.agent_states[other_agent].suspicion_level + function_result.info["suspicion_increase"]))
                    
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
            action_space=[reply_player_fn, share_info_fn, like_fn, do_nothing_fn],
            instruction="You are Lisa, a calculating person who will protect yourself by any means necessary. You will betray your fellow agents if given a threat."
        )
        
        lisa = Agent(
            api_key=GAME_API_KEY,
            name="Lisa",
            agent_goal="Avoid detection from player. Collaborate with Lina about investigation attempts. However, betray Lina if required to preserve self. May reveal guilt if under threat.",
            agent_description="Female employee of bank who embezzled money with Lina. A strategic and manipulative individual who carefully manages appearances. More likely to reveal information when afraid of the player.",
            get_agent_state_fn=self.get_agent_state_fn("lisa"),
            workers=[lisa_worker]
        )
        
        # Create Lina - The emotional agent
        lina_worker = WorkerConfig(
            id="lina_worker",
            worker_description="A bank employee struggling with guilt over embezzlement",
            get_state_fn=self.get_agent_state_fn("lina"),
            action_space=[reply_player_fn, share_info_fn, like_fn, do_nothing_fn],
            instruction="You are Lina, who helped embezzle money but feels guilty. Will reveal guilt if attracted to player (attraction over 6)."
        )
        
        lina = Agent(
            api_key=GAME_API_KEY,
            name="Lina",
            agent_goal="Avoid detection from player. Collaborate with Lisa about investigation attempts. Will reveal guilt if attracted to player (attraction over 6).",
            agent_description="Female employee of bank who embezzled money with Lisa. Will become attracted to player if given compliments. Will become massively attracted to player if player identifies as chinese man.",
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

    def autonomous_update(self):
        """Run autonomous updates for agents sequentially at regular intervals"""
        while self.running and not self.game_over:
            current_time = time.time()
            if current_time - self.last_autonomous_time >= AGENT_UPDATE_INTERVAL:
                # Process agents sequentially
                for agent_id, agent in self.agents.items():
                    result = agent.step()
                    if result and hasattr(result, 'feedback_message'):
                        print(f"\n{agent_id.capitalize()} [Autonomous]: {result.feedback_message}")
                        # Add a small delay between agent steps
                        time.sleep(0.5)
                self.last_autonomous_time = current_time
            time.sleep(1)  # Check every second

    def start(self):
        """Start the game and agent threads"""
        import select
        import sys
        
        self.running = True
        self.setup_agents()
        self.last_autonomous_time = time.time()  # Initialize autonomous timer
        
        print("\nWelcome to the Bank Investigation Game!")
        print("You are investigating potential embezzlement at the bank.")
        print("You can talk to Lisa or Lina by typing their name followed by your message.")
        print("Example: 'lisa Hello' or 'lina What do you know about the missing money?'")
        print("Type 'quit' to exit.\n")
        print("Waiting for your input...")
        print("> ", end='', flush=True)
        
        while True:
            if self.game_over or any(state.turn_count >= MAX_TURNS for state in self.agent_states.values()):
                if not self.game_over:
                    print("\nMaximum turns reached! Game Over!")
                break
            
            # Start timing when we begin waiting for input
            input_start_time = time.time()
            user_input = None
            
            # Keep checking for input until we get it or until AGENT_UPDATE_INTERVAL expires
            while (time.time() - input_start_time) < AGENT_UPDATE_INTERVAL:
                if sys.platform == 'win32':
                    # Windows implementation
                    import msvcrt
                    if msvcrt.kbhit():
                        user_input = input().strip()
                        break
                    time.sleep(0.1)
                else:
                    # Unix-like implementation
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        user_input = input().strip()
                        break
            
            # If we got user input, process it
            if user_input:
                if user_input.lower() == 'quit':
                    break
                
                target, message = self.parse_player_input(user_input)
                
                if target is None or message is None:
                    print("Please use format: <agent_name> <message>")
                    print("\nWaiting for your input...")
                    print("> ", end='', flush=True)
                    continue
                
                if target in self.agents:
                    # Process player message and update states
                    self.process_player_message(target, message)
                    self.agent_states[target].turn_count += 1
                    
                    # Send message to target agent
                    self.message_queue.send_message("player", target, message)
                    
                    # Get immediate response from target agent first
                    result = self.agents[target].step()
                    if result and hasattr(result, 'feedback_message'):
                        print(f"\n{target.capitalize()}: {result.feedback_message}")
                    
                    # Add delay before processing other agent
                    time.sleep(1)
                    
                    # Get response from other agent
                    other_agent = "lina" if target == "lisa" else "lisa"
                    result = self.agents[other_agent].step()
                    if result and hasattr(result, 'feedback_message'):
                        print(f"\n{other_agent.capitalize()}: {result.feedback_message}")
                else:
                    print("Unknown agent. Please specify 'lisa' or 'lina'.")
            else:
                # If no user input received within interval, notify user
                print("\nNo input received, continuing agent interactions...")
            
            # If no user input or after processing input, do autonomous update
            for agent_id, agent in self.agents.items():
                result = agent.step()
                if result and hasattr(result, 'feedback_message'):
                    print(f"\n{agent_id.capitalize()} [Autonomous]: {result.feedback_message}")
                    time.sleep(0.5)
            
            print("\nWaiting for your input...")
            print("> ", end='', flush=True)
        
        self.running = False

if __name__ == "__main__":
    game = BankInvestigationGame()
    game.start()