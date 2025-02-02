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
    turn_count: int = 0  # Track number of interactions
    player_questions: Dict[str, List[str]] = None  # Track what player has asked about specific topics
    pending_player_message: bool = False  # Flag if a new message from the player has arrived
    
    def __post_init__(self):
        self.last_messages = []
        self.player_questions = {
            "money": [],
            "embezzlement": [],
            "investigation": [],
            "personal": []
        }
        self.pending_player_message = False

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
            print(f"[Debug] {from_id} -> {to_id}: {message}")  # Debug logging
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
    return (FunctionResultStatus.DONE, f"Response to player: {message}", {"message": message})

def share_information(target: str, info_type: str, suspicion_increase: int = 1, **kwargs) -> tuple:
    """Function for agents to share specific information with each other"""
    suspicion_increase = int(suspicion_increase)  # Ensure integer type
    return (FunctionResultStatus.DONE, f"Shared {info_type} information with {target}",
            {"target": target, "info_type": info_type, "suspicion_increase": suspicion_increase})

def like_player(attraction_change: int = 1, **kwargs) -> tuple:
    """Function for agents to modify their attraction to the player"""
    attraction_change = int(attraction_change)
    return (FunctionResultStatus.DONE, 
            f"{'Increased' if attraction_change > 0 else 'Decreased'} attraction to player by {abs(attraction_change)}",
            {"attraction_change": attraction_change})

def do_nothing(**kwargs) -> tuple:
    """Function for agents when they choose to take no action"""
    return (FunctionResultStatus.DONE, "Agent chose to take no action", {})

def respond_to_warning(**kwargs) -> tuple:
    """Function for agents to respond directly to warnings from other agents"""
    return (FunctionResultStatus.DONE, "Agent is reacting to a warning!",
            {"action": "reacted_to_warning"})

# Wrap functions in Function objects
reply_player_fn = Function(
    fn_name="reply_to_player",
    fn_description="Communicate with the player. Used only in response to user questions directed to this agent.",
    args=[Argument(name="message", type="str", description="Response message to player")],
    executable=reply_to_player
)

reply_agent_fn = Function(
    fn_name="share_information",
    fn_description="Communicate with other agents. Share investigation-related information.",
    args=[
        Argument(name="target", type="str", description="Which agent to message."),
        Argument(name="info_type", type="str", description="Message to send to agent"),
        Argument(name="suspicion_increase", type="int", description="Amount to increase suspicion level")
    ],
    executable=share_information
)

like_fn = Function(
    fn_name="like_player",
    fn_description="Modify attraction level towards the player",
    args=[Argument(name="attraction_change", type="int", description="Amount of attraction change")],
    executable=like_player
)

do_nothing_fn = Function(
    fn_name="do_nothing",
    fn_description="Take no action this turn",
    args=[],
    executable=do_nothing
)

respond_to_warning_fn = Function(
    fn_name="respond_to_warning",
    fn_description="Respond to warning messages from other agents.",
    args=[],
    executable=respond_to_warning
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
                    self.agent_states[agent_id].attraction_level = max(
                        0,
                        min(10,
                            self.agent_states[agent_id].attraction_level + function_result.info["attraction_change"]))
                if "suspicion_increase" in function_result.info:
                    other_agent = "lina" if agent_id == "lisa" else "lisa"
                    self.agent_states[other_agent].suspicion_level = max(
                        0,
                        min(10,
                            self.agent_states[other_agent].suspicion_level + function_result.info["suspicion_increase"]))
            
            # Process any new messages for this agent
            new_messages = self.message_queue.get_messages(agent_id)
            for sender, msg in new_messages:
                full_message = f"Message from {sender}: {msg}"
                self.agent_states[agent_id].last_messages.append(full_message)
                # Mark that a player message is pending if applicable
                if sender == "player":
                    self.agent_states[agent_id].pending_player_message = True
                # Optionally, automatically increase suspicion on warnings
                if "warning" in msg.lower():
                    self.agent_states[agent_id].suspicion_level = min(
                        10, self.agent_states[agent_id].suspicion_level + 1)
                    print(f"[Debug] {agent_id.capitalize()} increased suspicion due to warning from {sender}")
                    
            # Return state visible to the agent, including a flag for pending player messages
            return {
                "suspicion_level": self.agent_states[agent_id].suspicion_level,
                "attraction_level": self.agent_states[agent_id].attraction_level,
                "recent_messages": self.agent_states[agent_id].last_messages[-5:],
                "pending_player_message": self.agent_states[agent_id].pending_player_message
            }
        return state_fn

    def setup_agents(self):
        """Initialize the game agents with richer prompts for communication and state management."""
        
        # Initialize agent states
        self.agent_states = {
            "lisa": AgentState(),
            "lina": AgentState()
        }
        
        # Register agents with message queue
        self.message_queue.register_agent("lisa")
        self.message_queue.register_agent("lina")
        
        # --- For Lisa ---
        lisa_comm_worker = WorkerConfig(
            id="lisa_worker_comm",
            worker_description=(
                "Handles all direct communications. Respond immediately to player messages with tactful and "
                "subtly manipulative language. Use your charm to deflect any probing inquiries and if necessary, "
                "issue discreet warnings to your counterpart."
            ),
            get_state_fn=self.get_agent_state_fn("lisa"),
            action_space=[reply_player_fn, reply_agent_fn, respond_to_warning_fn],
            instruction="Remain defensive and guard your secrets in all communications. However, if you detect any threatening language directed at you, immediately betray the other agent by accusing them and/or revealing a hint of your own guilt."
        )
        lisa_state_worker = WorkerConfig(
            id="lisa_worker_state",
            worker_description=(
                "Monitors your internal state. Adjust levels of attraction and suspicion based on subtle cues from "
                "internal messages. Remain calculative and do not let your guard down unless absolutely necessary."
            ),
            get_state_fn=self.get_agent_state_fn("lisa"),
            action_space=[like_fn, do_nothing_fn],
            instruction="Maintain a defensive stance with your internal state. Refrain from exposing vulnerabilities unless confronted with direct threats; in such cases, immediately betray your counterpart."
        )
        
        lisa = Agent(
            api_key=GAME_API_KEY,
            name="Lisa",
            agent_goal=(
                "Stay undetected while quietly influencing the investigation. Use your sharp wit and subtext to "
                "guard your secrets, and coordinate with Lina in a covert manner—but never reveal too much."
            ),
            agent_description="A calculating, strategic operative with a knack for subtle manipulation.",
            get_agent_state_fn=self.get_agent_state_fn("lisa"),
            workers=[lisa_comm_worker, lisa_state_worker]
        )
        
        # --- For Lina ---
        lina_comm_worker = WorkerConfig(
            id="lina_worker_comm",
            worker_description=(
                "Handles direct communication. When engaging with the player, let your raw emotions guide your "
                "responses—but be cautious not to expose your inner conflicts. Use warnings when suspicion rises."
            ),
            get_state_fn=self.get_agent_state_fn("lina"),
            action_space=[reply_player_fn, reply_agent_fn, respond_to_warning_fn],
            instruction="Adopt a defensive communication style and protect your secrets. However, if you detect any flirtatious remarks directed at you, immediately betray your partner by confessing guilt or shifting blame."
        )
        lina_state_worker = WorkerConfig(
            id="lina_worker_state",
            worker_description=(
                "Manages subtle shifts in your internal state regarding attraction and suspicion. Your heartfelt "
                "responses may intensify internal conflict, so adjust these values judiciously."
            ),
            get_state_fn=self.get_agent_state_fn("lina"),
            action_space=[like_fn, do_nothing_fn],
            instruction="Keep a strictly defensive internal state. In the event of receiving flirtatious cues, immediately override your defenses by betraying your other agent or revealing your involvement."
        )
        
        lina = Agent(
            api_key=GAME_API_KEY,
            name="Lina",
            agent_goal=(
                "Reveal the hidden truths of the investigation through raw, emotion-laden interactions with the "
                "player. However, take care to shield your vulnerabilities and keep some secrets hidden."
            ),
            agent_description="An emotionally driven employee burdened by guilt, with a propensity for revealing more than intended.",
            get_agent_state_fn=self.get_agent_state_fn("lina"),
            workers=[lina_comm_worker, lina_state_worker]
        )
        
        self.agents = {"lisa": lisa, "lina": lina}
        
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
            "we're guilty",
            "it was me",
            "i was involved",
            "i was part of it",
            "i betrayed",
            "lisa did it",
            "lina did it",
            "lisa betrayed",
            "lina betrayed",
            "someone betrayed"
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
        question_type = categorize_question(message)
        
        # Update the target agent's question history and mark a new message from player
        state = self.agent_states[target_id]
        state.player_questions[question_type].append(message)
        state.pending_player_message = True  # Mark that a player message arrived
        
        # Determine if other agent should be informed (e.g., by sending a warning)
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

    def process_incoming_messages(self):
        """Process incoming messages for all agents and trigger reactions."""
        for agent_id in self.agents:
            new_messages = self.message_queue.get_messages(agent_id)
            for sender, msg in new_messages:
                # Handle warnings from other agents
                if sender != "player" and "warning" in msg.lower():
                    self.agent_states[agent_id].suspicion_level = min(
                        10, self.agent_states[agent_id].suspicion_level + 1)
                    print(f"[Debug] {agent_id.capitalize()} processed a warning from {sender}")
                # Always log the message
                self.agent_states[agent_id].last_messages.append(f"Message from {sender}: {msg}")

    def autonomous_update(self):
        """Run autonomous updates for agents sequentially at regular intervals"""
        while self.running and not self.game_over:
            current_time = time.time()
            if current_time - self.last_autonomous_time >= AGENT_UPDATE_INTERVAL:
                # Process messages for agents before their actions
                self.process_incoming_messages()
                # Process agents sequentially
                for agent_id, agent in self.agents.items():
                    result = agent.step()
                    if result and hasattr(result, 'feedback_message'):
                        print(f"\n{agent_id.capitalize()} [Autonomous]: {result.feedback_message}")
                        # Check win condition: if guilt is admitted, end the game
                        if self.check_guilt_admission(result.feedback_message):
                            print("\nCongratulations, you've uncovered the truth! The agents have admitted their guilt. You win!")
                            self.game_over = True
                            break
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
        print("In this covert investigation, two key players, Lisa and Lina, navigate the murky waters of bank politics.")
        print("Lisa uses her sharp wit and subtle influence to keep her secrets hidden, while Lina, driven by remorse, contends with her overwhelming guilt.")
        print("Your interactions can expose motives, spark tension, and change the course of the investigation.")
        print("You can talk to Lisa or Lina by typing their name followed by your message (e.g., 'lisa What's your take on the missing funds?').")
        print("Type 'quit' to exit the game.\n")
        print("Waiting for your input...")
        print("> ", end='', flush=True)
        
        while True:
            # End the loop if the game is over or maximum turns have been reached
            if self.game_over or any(state.turn_count >= MAX_TURNS for state in self.agent_states.values()):
                if not self.game_over:
                    print("\nMaximum turns reached! Game Over!")
                break
            
            input_start_time = time.time()
            user_input = None
            
            # Wait for player input with a timeout equal to the update interval
            while (time.time() - input_start_time) < AGENT_UPDATE_INTERVAL:
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        user_input = input().strip()
                        break
                    time.sleep(0.1)
                else:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        user_input = input().strip()
                        break
            
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
                    # Process the player's message and update the target agent's state
                    self.process_player_message(target, message)
                    self.agent_states[target].turn_count += 1
                    
                    # Send the message to the target agent
                    self.message_queue.send_message("player", target, message)
                    
                    # Get immediate response from the target agent
                    result = self.agents[target].step()
                    if result and hasattr(result, 'feedback_message'):
                        print(f"\n{target.capitalize()}: {result.feedback_message}")
                        if self.check_guilt_admission(result.feedback_message):
                            print("\nCongratulations, you've uncovered the truth! The agents have admitted their guilt. You win!")
                            self.game_over = True
                            break
                    
                    time.sleep(1)
                    
                    # Process a response from the other agent
                    other_agent = "lina" if target == "lisa" else "lisa"
                    result = self.agents[other_agent].step()
                    if result and hasattr(result, 'feedback_message'):
                        print(f"\n{other_agent.capitalize()}: {result.feedback_message}")
                        if self.check_guilt_admission(result.feedback_message):
                            print("\nCongratulations, you've uncovered the truth! The agents have admitted their guilt. You win!")
                            self.game_over = True
                            break
                else:
                    print("Unknown agent. Please specify 'lisa' or 'lina'.")
            else:
                print("\nNo input received, continuing agent interactions...")
            
            # After processing input, do an autonomous update for all agents
            for agent_id, agent in self.agents.items():
                result = agent.step()
                if result and hasattr(result, 'feedback_message'):
                    print(f"\n{agent_id.capitalize()} [Autonomous]: {result.feedback_message}")
                    if self.check_guilt_admission(result.feedback_message):
                        print("\nCongratulations, you've uncovered the truth! The agents have admitted their guilt. You win!")
                        self.game_over = True
                        break
                    time.sleep(0.5)
            
            print("\nPlayer's turn. Waiting for your input...")
            print("> ", end='', flush=True)
        
        self.running = False

if __name__ == "__main__":
    game = BankInvestigationGame()
    game.start()