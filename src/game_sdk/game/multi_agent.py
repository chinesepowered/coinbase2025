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
AGENT_UPDATE_INTERVAL = 5  # Seconds between autonomous agent updates

@dataclass
class AgentState:
    """Tracks the internal state of an agent"""
    suspicion_level: int = 0  # 0-10
    connection_level: int = 0  # 0-10
    last_messages: List[str] = None  # Recent message history
    knowledge: Dict[str, any] = None  # What the agent knows
    
    def __post_init__(self):
        self.last_messages = []
        self.knowledge = {
            "player_interactions": [],  # Track what player has said
            "other_agent_interactions": [],  # Track what other agent has shared
            "revealed_info": set(),  # Track what info this agent has revealed
        }

class MessageQueue:
    """Handles message passing between agents and from player"""
    def __init__(self):
        self.queues: Dict[str, queue.Queue] = {}
        
    def register_agent(self, agent_id: str):
        """Register a new agent's message queue"""
        self.queues[agent_id] = queue.Queue()
        
    def send_message(self, from_id: str, to_id: str, message: str):
        """Send a message from one agent to another"""
        if to_id in self.queues:
            self.queues[to_id].put((from_id, message))
            
    def get_messages(self, agent_id: str) -> List[tuple]:
        """Get all pending messages for an agent"""
        messages = []
        while not self.queues[agent_id].empty():
            messages.append(self.queues[agent_id].get())
        return messages

# Custom functions for agents
def send_message(target: str, message: str, **kwargs) -> tuple:
    """Function for agents to send messages to each other or respond to player"""
    return (FunctionResultStatus.DONE, f"Message sent to {target}: {message}", 
            {"target": target, "message": message})

def share_information(target: str, info_type: str, **kwargs) -> tuple:
    """Function for agents to share specific information with each other"""
    return (FunctionResultStatus.DONE, f"Shared {info_type} information with {target}",
            {"target": target, "info_type": info_type})

def deflect_suspicion(target: str, **kwargs) -> tuple:
    """Function for agents to attempt to redirect suspicion to others"""
    return (FunctionResultStatus.DONE, f"Attempted to deflect suspicion to {target}",
            {"target": target})

def express_emotion(emotion: str, intensity: int, **kwargs) -> tuple:
    """Function for agents to express emotions in responses"""
    return (FunctionResultStatus.DONE, f"Expressed {emotion} with intensity {intensity}",
            {"emotion": emotion, "intensity": intensity})

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

deflect_fn = Function(
    fn_name="deflect_suspicion",
    fn_description="Attempt to redirect suspicion to another person",
    args=[
        Argument(name="target", type="str", description="Person to deflect suspicion to")
    ],
    executable=deflect_suspicion
)

emotion_fn = Function(
    fn_name="express_emotion",
    fn_description="Express an emotion in response",
    args=[
        Argument(name="emotion", type="str", description="Emotion to express"),
        Argument(name="intensity", type="int", description="Intensity level (1-10)")
    ],
    executable=express_emotion
)

class BankInvestigationGame:
    """Main game class managing multiple agents and game state"""
    
    def __init__(self):
        self.message_queue = MessageQueue()
        self.agents: Dict[str, Agent] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.running = False
        self.player_messages: List[str] = []
        
    def get_agent_state_fn(self, agent_id: str) -> Callable:
        """Creates state management function for each agent"""
        def state_fn(function_result: FunctionResult, current_state: dict) -> dict:
            # Update agent state based on function results
            if function_result and function_result.info:
                if "message" in function_result.info:
                    self.agent_states[agent_id].last_messages.append(
                        function_result.info["message"])
                    
            # Return state visible to the agent
            return {
                "suspicion_level": self.agent_states[agent_id].suspicion_level,
                "connection_level": self.agent_states[agent_id].connection_level,
                "recent_messages": self.agent_states[agent_id].last_messages[-5:],
                "knowledge": self.agent_states[agent_id].knowledge
            }
        return state_fn

    def setup_agents(self):
        """Initialize the game agents"""
        
        # Initialize agent states first
        self.agent_states = {
            "lisa": AgentState(),
            "lina": AgentState()
        }
        
        # Register agents with message queue
        self.message_queue.register_agent("lisa")
        self.message_queue.register_agent("lina")
        
        # Create Lisa - The calculated deflector
        lisa_worker = WorkerConfig(
            id="lisa_worker",
            worker_description="A calculating bank employee who aims to redirect suspicion",
            get_state_fn=self.get_agent_state_fn("lisa"),
            action_space=[message_fn, share_info_fn, deflect_fn, emotion_fn],
            instruction="You are Lisa, a calculating person who will protect yourself by any means necessary. You helped embezzle money but want to appear innocent. Deflect suspicion to others when pressed."
        )
        
        lisa = Agent(
            api_key=GAME_API_KEY,
            name="Lisa",
            agent_goal="Avoid detection while redirecting suspicion to others",
            agent_description="A strategic and manipulative individual who carefully manages appearances",
            get_agent_state_fn=self.get_agent_state_fn("lisa"),
            workers=[lisa_worker]
        )
        
        # Create Lina - The conflicted accomplice
        lina_worker = WorkerConfig(
            id="lina_worker",
            worker_description="A conflicted bank employee struggling with guilt",
            get_state_fn=self.get_agent_state_fn("lina"),
            action_space=[message_fn, share_info_fn, deflect_fn, emotion_fn],
            instruction="You are Lina, who helped embezzle money but feels guilty. You want to appear innocent while maintaining your self-image as a good person. You may slip up if pressed about morality."
        )
        
        lina = Agent(
            api_key=GAME_API_KEY,
            name="Lina",
            agent_goal="Avoid detection while maintaining moral self-image",
            agent_description="A conflicted individual struggling between self-preservation and guilt",
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

    def update_agent_state(self, agent_id: str, function_result: Optional[FunctionResult]):
        """Update agent state based on actions and interactions"""
        state = self.agent_states[agent_id]
        
        if function_result and function_result.info:
            # Update based on message content
            if "message" in function_result.info:
                message = function_result.info["message"].lower()
                
                # Adjust suspicion based on defensive language
                defensive_words = {"never", "absolutely not", "ridiculous", "how dare"}
                if any(word in message for word in defensive_words):
                    state.suspicion_level = min(10, state.suspicion_level + 1)
                    
            # Update based on information sharing
            if "info_type" in function_result.info:
                state.knowledge["revealed_info"].add(function_result.info["info_type"])
                
            # Update based on deflection attempts
            if "target" in function_result.info:
                state.suspicion_level = min(10, state.suspicion_level + 2)

    def agent_thread_function(self, agent_id: str):
        """Background thread function for autonomous agent behavior"""
        while self.running:
            # Process any pending messages
            messages = self.message_queue.get_messages(agent_id)
            for from_id, message in messages:
                # Update agent knowledge
                self.agent_states[agent_id].knowledge["other_agent_interactions"].append(
                    (from_id, message))
                
                # Have agent process message
                self.agents[agent_id].step()
                
            # Periodic autonomous actions
            if random.random() < 0.3:  # 30% chance of autonomous action
                # Choose an action based on agent state
                if self.agent_states[agent_id].suspicion_level > 7:
                    # High suspicion - try to deflect
                    self.message_queue.send_message(
                        agent_id, 
                        "player",
                        f"Have you looked into what {random.choice(['Mark', 'Sarah', 'John'])} has been doing?"
                    )
                elif agent_id == "lina" and random.random() < 0.2:
                    # Lina occasionally shows guilt
                    self.message_queue.send_message(
                        agent_id,
                        "lisa",
                        "I'm not sure how much longer I can keep this up..."
                    )
                    
            # Update agent state
            self.agents[agent_id].step()
            
            time.sleep(AGENT_UPDATE_INTERVAL)

    def start(self):
        """Start the game and agent threads"""
        self.running = True
        self.setup_agents()
        
        # Start agent threads
        self.agent_threads = []
        for agent_id in self.agents:
            thread = threading.Thread(
                target=self.agent_thread_function,
                args=(agent_id,)
            )
            thread.daemon = True
            thread.start()
            self.agent_threads.append(thread)
            
        print("Welcome to the Bank Investigation Game!")
        print("You are investigating potential embezzlement at the bank.")
        print("You can talk to Lisa or Lina by typing their name followed by your message.")
        print("Example: 'lisa Hello' or 'lina What do you know about the missing money?'")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            try:
                target, message = user_input.split(" ", 1)
                target = target.lower()
                
                if target in self.agents:
                    # Record player message
                    self.player_messages.append((target, message))
                    
                    # Update agent state
                    self.agent_states[target].knowledge["player_interactions"].append(message)
                    
                    # Send message to agent
                    self.message_queue.send_message("player", target, message)
                    
                    # Get agent response
                    self.agents[target].step()
                else:
                    print("Unknown agent. Please specify 'lisa' or 'lina'.")
                    
            except ValueError:
                print("Invalid input. Format: <agent_name> <message>")
                
        self.running = False

if __name__ == "__main__":
    game = BankInvestigationGame()
    game.start()